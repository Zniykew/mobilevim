# post_pretrain.py
import os
import sys
import argparse
import datetime
import random
import json
import gc
import time

import math
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import Accuracy
from PIL import Image
import matplotlib.pyplot as plt

import model.mobilevim2

def setup_distributed():
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=datetime.timedelta(seconds=30)
    )
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    return local_rank

class ImageNetConfig:
    data_root = "/workspace/datasets/ILSVRC2012"
    train_dir = "train"
    val_dir = "val"
    model_type = "xx_small"
    num_classes = 1000
    dropout = 0.3
    epochs = 180
    warmup_epochs = 10
    batch_size = 128
    base_lr = 6e-4
    weight_decay = 1e-4
    grad_accum_steps = 2
    aa_magnitude = 9
    re_prob = 0.2
    interpolation = transforms.InterpolationMode.BILINEAR
    im2col_step = 64
    patience = 20
    mid_delta = 0.002
    min_lr = 1e-6  # 最低学习率
    decay_strategy = "linear"  # 使用线性衰减

class ImageNetTrainer:
    def __init__(self, config, pretrained_path, local_rank=-1):
        self.config = config
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.device = torch.device(f"cuda:{local_rank}" if self.is_distributed else "cuda:0")
        self.best_val_metric = -float("inf")
        self.early_stop_counter = 0
        self.warmup_epochs = 5
        self.total_steps = 0  # 新增全局step计数器
        self.steps_per_epoch = 2500 // self.config.grad_accum_steps  # 1250 steps/epoch

        backbone = model.mobilevim2.MobileViM(
            num_classes=0,  # 禁用内部分类头
            model_type=config.model_type,
            in_chans=3,
            img_size=224
        ).to(self.device)

        # 添加自定义分类头
        self.classifier = model.mobilevim2.ClassificationHead(
            in_channels=256,  # 匹配MobileViM的decoder隐藏维度
            num_classes=config.num_classes,
            dropout=config.dropout
        ).to(self.device)

        self.model = nn.Sequential(backbone, self.classifier)
        self.train_loader, self.val_loader = self._init_dataloaders()

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda  # 直接传递方法
        )
        self.scaler = GradScaler(enabled=True)

        self.top1_acc = Accuracy(task='multiclass', num_classes=1000, top_k=1).to(self.device)
        self.top5_acc = Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.device)
        if self.local_rank <= 0:
            Path("/workspace/zyw/MobileVim/imagenet_log").mkdir(parents=True, exist_ok=True)
            Path("/workspace/zyw/MobileVim/imagenet_run").mkdir(parents=True, exist_ok=True)

        # 加载预训练模型
        self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        pretrained_dict = torch.load(path, map_location='cpu')
        # 处理分布式训练保存的键名前缀
        if any(k.startswith('module.') for k in pretrained_dict.keys()):
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}

        # 移除自定义分类头的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('classifier.')}

        # 加载模型参数
        self.model.load_state_dict(pretrained_dict, strict=False)
        if self.local_rank <= 0:
            print(f"Successfully loaded pretrained weights.")

    def _lr_lambda(self, current_step):
        warmup_steps = self.config.warmup_epochs * self.steps_per_epoch
        total_steps = self.config.epochs * self.steps_per_epoch

        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return max(0.0, 1.0 - (current_step - warmup_steps) / (total_steps - warmup_steps))

    def _init_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=self.config.interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=3,  # 默认操作次数
                magnitude=self.config.aa_magnitude,  # magnitude=9
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=self.config.re_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=self.config.interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageFolder(os.path.join(self.config.data_root, self.config.train_dir), transform=train_transform)
        val_dataset = ImageFolder(os.path.join(self.config.data_root, self.config.val_dir), transform=val_transform)
        train_sampler = DistributedSampler(train_dataset) if self.is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.is_distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.top1_acc.reset()
        self.top5_acc.reset()
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=True):
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)

            self.scaler.scale(loss).backward()

            # 梯度累积逻辑
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # 更新学习率
                self.total_steps += 1
                self.scheduler.step()

            total_loss += loss.item()
            self.top1_acc.update(outputs, labels)
            self.top5_acc.update(outputs, labels)

            if (batch_idx % 50 == 0) and (self.local_rank == 0):
                samples_sec = self.config.batch_size * (batch_idx + 1) / (time.time() - start_time)
                print(f"Epoch {epoch} | Batch {batch_idx:04d} | Loss: {loss.item():.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f} | Speed: {samples_sec:.1f} samples/s")
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

        return {
            'train_loss': total_loss / len(self.train_loader),
            'top1_acc': self.top1_acc.compute().item(),
            'top5_acc': self.top5_acc.compute().item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        self.top1_acc.reset()
        self.top5_acc.reset()
        start_time = time.time()  # 新增
        for batch_idx, (images, labels) in enumerate(self.val_loader):  # 修改为enumerate
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            self.top1_acc.update(outputs, labels)
            self.top5_acc.update(outputs, labels)

            # 每50个batch输出
            if (batch_idx % 50 == 0) and (self.local_rank <= 0):
                avg_loss = total_loss / (batch_idx + 1)
                samples_sec = 64 * (batch_idx + 1) / (time.time() - start_time)  # val batch_size=64
                print(f"Validation | Batch {batch_idx:04d} | Loss: {avg_loss:.4f} | Speed: {samples_sec:.1f} samples/s")

        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_top1': self.top1_acc.compute().item(),
            'val_top5': self.top5_acc.compute().item()
        }

    def run(self):
        best_acc = 0.0
        log_df = pd.DataFrame()
        log_path = Path("/workspace/zyw/MobileVim/imagenet_log/training_log.txt")

        # 日志文件初始化
        if self.local_rank <= 0 and not log_path.exists():
            with log_path.open("w") as f:
                f.write("epoch\tlr\ttrain_loss\ttop1_acc\ttop5_acc\tval_loss\tval_top1\n")

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            log_entry = {'epoch': epoch, **train_metrics, **val_metrics}
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)

            # 检查是否需要保存模型
            if val_metrics['val_top1'] > self.best_val_metric + self.config.mid_delta:
                self.best_val_metric = val_metrics['val_top1']
                self.early_stop_counter = 0  # 重置计数器

                if not self.is_distributed or self.local_rank == 0:
                    model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
                    save_path = Path(
                        "/workspace/zyw/MobileVim/weights") / f"mobilevim_imagenet_top1_{self.best_val_metric:.3f}.pth"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model_state, save_path)
            else:
                self.early_stop_counter += 1  # 增加计数器

                # 检查是否触发早停
                if self.early_stop_counter >= self.config.patience:
                    if self.local_rank <= 0:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break  # 提前终止训练

                # 保存日志
                if not self.is_distributed or self.local_rank == 0:
                    log_df.to_csv("imagenet_training_log.csv", index=False)

            if self.local_rank <= 0:
                print(f"Epoch {epoch} Summary | "
                      f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Top1 Acc: {train_metrics['top1_acc']:.4f} | "
                      f"Top5 Acc: {train_metrics['top5_acc']:.4f} | "
                      f"Val Top1: {val_metrics['val_top1']:.4f}")
                log_line = (f"{epoch}\t{train_metrics['lr']:.6f}\t{train_metrics['train_loss']:.4f}\t"
                            f"{train_metrics['top1_acc']:.4f}\t{train_metrics['top5_acc']:.4f}\t"
                            f"{val_metrics['val_loss']:.4f}\t{val_metrics['val_top1']:.4f}\n")
                with log_path.open("a") as f:
                    f.write(log_line)

        if self.local_rank <= 0:
            self._generate_plots(log_df)

        return best_acc

    def _generate_plots(self, log_df):
        """生成训练指标可视化图表"""
        plt.figure(figsize=(14, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(log_df['epoch'], log_df['train_loss'], label='Train')
        plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # 准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(log_df['epoch'], log_df['top1_acc'], label='Top1 Train')
        plt.plot(log_df['epoch'], log_df['val_top1'], label='Top1 Val')
        plt.plot(log_df['epoch'], log_df['top5_acc'], label='Top5 Train')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        # 学习率曲线
        plt.subplot(2, 2, 3)
        plt.plot(log_df['epoch'], log_df['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True)
        plt.legend()

        # 综合指标
        plt.subplot(2, 2, 4)
        plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
        plt.plot(log_df['epoch'], log_df['val_top1'], label='Val Top1')
        plt.title('Training Correlation')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("/workspace/zyw/MobileVim/imagenet_run/training_metrics.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to the pretrained model checkpoint")
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1))
    args = parser.parse_args()

    local_rank = args.local_rank
    if local_rank >= 0:
        local_rank = setup_distributed()

    config = ImageNetConfig()
    trainer = ImageNetTrainer(config, args.pretrained_path, local_rank)
    best_acc = trainer.run()
    if local_rank <= 0:
        print(f"Post-pretraining completed with best accuracy: {best_acc:.2f}%")
