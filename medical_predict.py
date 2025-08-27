import os
import sys
import argparse
import datetime
import random
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib.pyplot as plt

# 导入数据加载函数
from med_dataloader.test_medDataloader import create_data_loaders

# 确保使用与文档1相同的模型架构
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

# -------------------- 医疗训练配置 --------------------
class MedicalConfig:
    # 数据集配置
    dataset = 'smalldata'  # 指定数据集类型，如'smalldata'、'blender'或'colon'
    data_root = "/data/dataset/Medical"  # 根据实际路径调整
    max_depth = 20.0  # 根据数据集调整
    batch_size = 128
    epochs = 150
    base_lr = 1e-4
    weight_decay = 1e-5
    grad_accum_steps = 2
    max_grad_norm = 8.0
    log_interval = 50
    num_workers = 4
    pretrained_path = "/workspace/zyw/MobileVim/weights/depth_model_loss_0.1770.pth"
    output_dir = "/workspace/zyw/MobileVim/medical_results/smalldata"
    patience = 15
    min_lr = 1e-6
    decay_strategy = "cosine"
    model_type = "xx_small"

    @property
    def img_size(self):
        """根据数据集类型动态返回图像尺寸"""
        size_map = {
            'smalldata': 320,
            'blender': 320,
            'colon': 256
        }
        return size_map[self.dataset]

# -------------------- 训练器 --------------------
class MedicalTrainer:
    def __init__(self, config, local_rank=-1):
        self.config = config
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.device = torch.device(f"cuda:{local_rank}" if self.is_distributed else "cuda:0")
        self.best_metric = float('inf')
        self.early_stop_counter = 0
        self.effective_batch_size = None

        # 构建模型
        self.backbone = model.mobilevim2.MobileViM(
            num_classes=0,
            model_type=config.model_type,
            in_chans=3,
            img_size=config.img_size
        ).to(self.device)
        self.decoder = model.mobilevim2.DepthPostProcess(
            in_channels=256,
            upscale_factor=4
        ).to(self.device)
        self.model = nn.Sequential(self.backbone, self.decoder)
        self._load_pretrained()

        # 冻结backbone前三个阶段
        for name, param in self.backbone.named_parameters():
            if 'stage' in name and int(name.split('.')[1]) < 3:
                param.requires_grad_(False)

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank])

        self.train_loader, self.val_loader = self._init_dataloaders()
        # 优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 1.0
        )
        self.scaler = GradScaler(enabled=True)


        if self.local_rank <= 0:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _lr_lambda(self, current_step):
        """动态调整学习率策略"""
        warmup_steps = int(0.1 * self.config.epochs * len(self.train_loader))  # 10%的epoch用于warmup
        total_steps = self.config.epochs * len(self.train_loader)

        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (
                        1 - self.config.min_lr / self.config.base_lr) + self.config.min_lr / self.config.base_lr
    def _load_pretrained(self):
        try:
            # 加载完整预训练权重
            state_dict = torch.load(self.config.pretrained_path, map_location='cpu')

            # 兼容不同保存格式的键名转换
            new_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                # 兼容新旧版本结构差异
                if k.startswith("backbone."):
                    new_key = f"0.{k[9:]}"  # 适配Sequential结构
                elif k.startswith("decoder."):
                    new_key = f"1.{k[7:]}"
                else:
                    new_key = k
                new_dict[new_key] = v

            # 加载并允许部分权重不匹配（如分类头）
            missing, unexpected = self.model.load_state_dict(new_dict, strict=False)

            if self.local_rank <= 0:
                print(f"成功加载预训练权重 from {self.config.pretrained_path}")
                if missing:
                    print(f"未找到对应参数: {missing[:5]}{'...' if len(missing) > 5 else ''}")
                if unexpected:
                    print(f"冗余参数: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        except Exception as e:
            if self.local_rank <= 0:
                print(f"加载预训练模型失败: {str(e)}")
                print("请检查：")
                print("1. 预训练路径是否正确")
                print("2. 模型结构是否与预训练权重匹配")
                print("3. 分布式训练保存的权重是否包含多余前缀")
            raise RuntimeError("预训练权重加载失败") from e

    def _init_dataloaders(self):
        class Args:
            def __init__(self, config):
                self.dataset = config.dataset
                self.batch_size = config.batch_size
                self.workers = config.num_workers
                self.max_depth = config.max_depth
                self.img_size = config.img_size

        args = Args(self.config)

        # 获取原始DataLoader并提取Dataset
        train_loader_orig, val_loader_orig = create_data_loaders(args)
        train_dataset = train_loader_orig.dataset
        val_dataset = val_loader_orig.dataset

        # 分布式参数
        world_size = dist.get_world_size() if self.is_distributed else 1
        self.effective_batch_size = self.config.batch_size // world_size

        # 创建采样器
        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)

        # 创建新的DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.effective_batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.effective_batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0
        )
        return train_loader, val_loader

    def run(self):
        best_loss = float('inf')
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            # 只在主进程处理日志和保存
            if self.local_rank <= 0:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # 保存最佳模型
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self.early_stop_counter = 0
                    self._save_model(epoch)
                else:
                    self.early_stop_counter += 1

                # 早停检查
                if self.early_stop_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.best_metric

    def _save_model(self, epoch):
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        save_path = Path(self.config.output_dir) / f"best_model_epoch{epoch}_loss{self.best_metric:.4f}.pth"
        torch.save(model_state, save_path)
        print(f"Saved best model to {save_path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 添加内存监控
        torch.cuda.empty_cache()
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            depths = batch['depth'].to(self.device, non_blocking=True)
            images = (images - mean) / std

            with autocast(enabled=True):
                preds = self.model(images)
                preds = F.interpolate(preds, size=depths.shape[-2:], mode='bilinear')
                valid_mask = (depths > 0.01) & (depths < self.config.max_depth)
                loss = F.l1_loss(preds[valid_mask], depths[valid_mask])

            # 梯度累积优化
            loss = loss / self.config.grad_accum_steps
            self.scaler.scale(loss).backward()

            # 梯度更新条件判断
            if (batch_idx + 1) % self.config.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # 使用更高效的内存释放方式
                self.scheduler.step()

            total_loss += loss.item() * self.config.grad_accum_steps  # 恢复实际损失值

            # 添加内存监控日志
            if batch_idx % self.config.log_interval == 0 and self.local_rank <= 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                samples_sec = self.effective_batch_size * (batch_idx + 1) / (time.time() - start_time)
                print(
                    f"Epoch {epoch} | Batch {batch_idx:04d} | Loss: {loss.item() * self.config.grad_accum_steps:.4f} | "
                    f"Mem: {mem:.2f}GB | Speed: {samples_sec:.1f} samples/s")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            images = (images - mean) / std  # 应用归一化

            preds = self.model(images)
            preds = F.interpolate(preds, size=depths.shape[-2:], mode='bilinear')
            valid_mask = (depths > 0.01) & (depths < self.config.max_depth)
            loss = F.l1_loss(preds[valid_mask], depths[valid_mask])
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        local_rank = setup_distributed()

    config = MedicalConfig()
    trainer = MedicalTrainer(config, local_rank)
    best_loss = trainer.run()

    if local_rank <= 0:
        print(f"训练完成，最佳验证损失: {best_loss:.4f}")