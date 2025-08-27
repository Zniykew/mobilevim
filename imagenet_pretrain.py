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
    dropout = 0.2
    epochs = 300
    warmup_epochs = 20
    batch_size = 128
    base_lr = 1e-4
    weight_decay = 1e-5
    grad_accum_steps = 1
    decay_rate = 0.97
    decay_epochs = 2.4
    aa_magnitude = 9
    aa_std = 0.5
    re_prob = 0.2
    interpolation = transforms.InterpolationMode.BILINEAR
    im2col_step = 64
    patience = 20
    mid_delta = 0.002
    min_lr = 1e-6  # 最低学习率
    decay_strategy = "cosine"  # 使用余弦退火

class ImageNetTrainer:
    def __init__(self, config, local_rank=-1):
        self.config = config
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.device = torch.device(f"cuda:{local_rank}" if self.is_distributed else "cuda:0")
        self.best_val_metric = -float("inf")
        self.early_stop_counter = 0
        self.warmup_epochs = 5
        self.total_steps = 0  # 全局step计数器
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
            weight_decay=config.weight_decay
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

    def _lr_lambda(self, current_step):
        warmup_steps = self.config.warmup_epochs * self.steps_per_epoch
        total_steps = self.config.epochs * self.steps_per_epoch

        if current_step < warmup_steps:
            # Warmup阶段线性增长
            return current_step / warmup_steps
        else:
            # 余弦退火阶段
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (
                        1 - self.config.min_lr / self.config.base_lr) + self.config.min_lr / self.config.base_lr

    def _init_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=self.config.interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=2,  # 默认操作次数
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
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            sampler=val_sampler,
            num_workers=2,
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
class DepthConfig:
    model_type = "xx_small"
    datasets = {
        'nyu': {
            'data_root': "/workspace/datasets/NYU_Depth_V2",
            'depth_scale': 1000.0,
            'max_depth': 10.0,
            'input_size': (480, 640)
        },
        'kitti': {
            'data_root': "/workspace/datasets/KITTI",
            'depth_scale': 256.0,
            'max_depth': 80.0,
            'input_size': (384, 1280)
        }
    }

    # 公共基础配置（可被覆盖）
    epochs = 200
    base_lr = 1e-4
    weight_decay = 1e-5
    grad_accum_steps = 2
    max_grad_norm = 8.0
    log_interval = 50

    # NYU 专用配置
    nyu_batch_size = 64
    nyu_num_workers = 8
    nyu_input_size = (224, 224)

    # KITTI 专用配置
    kitti_batch_size = 8   # 减小 batch size
    kitti_num_workers = 4   # 减少 worker 数量
    kitti_input_size = (320, 1024)  # 更大输入尺寸



class DepthDataset(Dataset):
    def __init__(self, mode='train', dataset='nyu'):
        """
        NYU Depth V2 或 KITTI 深度估计数据集
        参数:
            mode: 'train' 或 'val'
            dataset: 'nyu' 或 'kitti'
        """
        self.config = DepthConfig.datasets[dataset]
        self.mode = mode
        self.dataset = dataset

        # 基础路径设置
        if dataset == 'nyu':
            if mode == 'train':
                self.base_path = Path("/workspace/datasets/NYU_Depth_V2/sync")
            else:
                self.base_path = Path("/workspace/datasets/NYU_Depth_V2/test")
        elif dataset == 'kitti':
            self.base_path = Path("/workspace/datasets/KITTI")

        # 构建分割文件路径
        if dataset == 'kitti':
            split_name = {
                'train': 'kitti_eigen_train.txt',
                'val': 'kitti_eigen_test.txt',
                'test': 'kitti_eigen_test.txt'
            }[mode]
            split_file = os.path.join("/workspace/datasets/KITTI", split_name)
        else:
            split_file = os.path.join(self.config['data_root'], f"{dataset}_{mode}.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found")

        # 读取并验证数据样本
        self.samples = []
        with open(split_file, 'r') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 2 and len(parts) != 3:  # 兼容只有两个字段的情况
                    raise ValueError(f"Invalid line format at line {line_idx}: {line}")

                rgb_rel = parts[0]
                depth_rel = parts[1]

                # 新增：如果 depth_rel 是 "None" 或空字符串，则跳过该样本
                if not depth_rel or depth_rel.lower() == 'none':
                    continue

                rgb_path = self.base_path / rgb_rel
                if dataset == 'nyu':
                    depth_path = self.base_path / depth_rel
                elif dataset == 'kitti':
                    depth_path = Path(self.base_path) / "data_depth_annotated" / depth_rel

                if not rgb_path.exists():
                    print(f"RGB image not found: {rgb_path} (line {line_idx}), skipped.")
                    continue
                if not depth_path.exists():
                    print(f"Dense depth map not found: {depth_path} (line {line_idx}), skipped.")
                    continue

                self.samples.append((str(rgb_path), str(depth_path)))


        # 数据增强配置
        self._init_transforms()

    def _init_transforms(self):
        if self.dataset == 'nyu':
            input_size = (224, 224)
        elif self.dataset == 'kitti':
            input_size = (384, 1280)

        if self.mode == 'train':
            self.rgb_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.depth_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.5, 1.0),
                    interpolation=Image.NEAREST
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.depth_transform = transforms.Compose([
                transforms.Resize(
                    input_size,
                    interpolation=Image.NEAREST
                ),
                transforms.CenterCrop(input_size),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回:
            tuple: (rgb_tensor, depth_tensor)
                  深度值已归一化到[0, max_depth]
        """
        rgb_path, depth_path = self.samples[idx]

        try:
            # 加载RGB图像
            rgb_img = Image.open(rgb_path).convert('RGB')

            # 加载深度图并转换格式
            depth_img = Image.open(depth_path)
            if self.dataset == 'nyu':
                depth_array = np.asarray(depth_img, dtype=np.float32) / self.config['depth_scale']
                depth_array = np.clip(depth_array, 0, self.config['max_depth'])
                depth_pil = Image.fromarray(depth_array)
            else:  # KITTI使用16位PNG
                # 原始深度图为 16-bit PNG，每个像素值表示毫米级距离
                depth_array = np.asarray(depth_img, dtype=np.float32)
                depth_array = depth_array / self.config['depth_scale']  # 转换为米单位
                depth_array = np.clip(depth_array, 0, self.config['max_depth'])
                depth_pil = Image.fromarray(depth_array)

            # 训练模式数据增强
            if self.mode == 'train':
                # 随机几何变换（统一参数）
                seed = np.random.randint(2147483647)

                # RGB变换
                random.seed(seed)
                torch.manual_seed(seed)
                rgb_tensor = self.rgb_transform(rgb_img)

                # 深度变换（同步参数）
                random.seed(seed)
                torch.manual_seed(seed)
                depth_pil = self.depth_transform(depth_pil)
            else:
                rgb_tensor = self.rgb_transform(rgb_img)
                depth_pil = self.depth_transform(depth_pil)

                # 转换深度图格式
            depth_tensor = torch.from_numpy(np.array(depth_pil, dtype=np.float32)).unsqueeze(0)

            return rgb_tensor, depth_tensor

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # 返回空数据避免训练中断
            dummy_input = torch.zeros(3, *self.config['input_size'])
            dummy_depth = torch.zeros(1, *self.config['input_size'])
            return dummy_input, dummy_depth
class DepthTrainer:
    def __init__(self, config, pretrained_path, local_rank=-1, dataset='nyu'):
        self.config = config
        self.local_rank = local_rank
        self.dataset = dataset
        self.is_distributed = local_rank >= 0
        self.device = torch.device(f"cuda:{local_rank}" if self.is_distributed else "cuda:0")

        self.backbone = model.mobilevim2.MobileViM(
            num_classes=0,
            model_type=config.model_type,
            in_chans=3,
            img_size=224
        ).to(self.device)

        self._load_pretrained(pretrained_path)

        self.decoder = model.mobilevim2.DepthPostProcess(
            in_channels=256,
            upscale_factor=4
        ).to(self.device)

        self.model = nn.Sequential(self.backbone, self.decoder)

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, static_graph=True)

        # 优化器仅训练未冻结层和decoder
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )
        self.scaler = GradScaler(enabled=True)
        self.train_loader = self._init_dataloader()
        self.val_loader = self._init_val_dataloader()

    def _load_pretrained(self, path):
        pretrained_dict = torch.load(path, map_location='cpu')
        # 处理分布式训练保存的键名前缀
        if any(k.startswith('module.') for k in pretrained_dict.keys()):
            pretrained_dict = {k.replace('module.0.', ''): v for k, v in pretrained_dict.items()
                               if k.startswith('module.0.')}  # 提取backbone参数
        else:
            pretrained_dict = {k[2:]: v for k, v in pretrained_dict.items()
                               if k.startswith('0.')}  # 普通训练保存的格式

        # 加载backbone参数
        self.backbone.load_state_dict(pretrained_dict, strict=True)

        # 冻结前三阶段的参数
        for name, param in self.backbone.named_parameters():
            if 'stage' in name and int(name.split('.')[1]) < 3:
                param.requires_grad_(False)
        if self.local_rank <= 0:
            print(f"Successfully loaded pretrained backbone weights. Frozen stages 0-2.")

    def _init_dataloader(self):
        dataset = DepthDataset(dataset=self.dataset)
        sampler = DistributedSampler(dataset) if self.is_distributed else None
        batch_size = self.config.kitti_batch_size if self.dataset == 'kitti' else self.config.nyu_batch_size
        num_workers = self.config.kitti_num_workers if self.dataset == 'kitti' else self.config.nyu_num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    def _init_val_dataloader(self):
        val_dataset = DepthDataset(mode='test', dataset=self.dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.is_distributed else None
        batch_size = 16 if self.dataset == 'kitti' else 32
        num_workers = 2 if self.dataset == 'kitti' else 4
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

    @staticmethod
    def berhu_loss(pred, target, C=None):
        """
        BerHu Loss (Reverse Huber Loss)

        Args:
            pred: 预测深度图 (Tensor)
            target: 真实深度图 (Tensor)
            C: 超参数，通常为最大绝对误差的 1/5，默认为两者差值的 1/5

        Returns:
            loss: BerHu Loss 值
        """
        abs_error = torch.abs(pred - target)
        if C is None:
            C = 0.2 * torch.max(abs_error).item()  # 可根据数据调整
        quadratic = torch.clamp(abs_error, max=C)
        linear = (abs_error - C).clamp(min=0)
        loss = torch.mean(0.5 * (quadratic ** 2) / C + linear)
        return loss


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        metric_sum = {
            "delta1": 0.0, "delta2": 0.0, "delta3": 0.0,
            "rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0, "silog": 0.0
        }

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, depths) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            depths = depths.to(self.device, non_blocking=True)

            with autocast(enabled=True):
                preds = self.model(images)
                preds = F.interpolate(preds, size=depths.shape[-2:], mode='bilinear')
                valid_mask = (depths > 0.01) & (depths < self.config.datasets['nyu']['max_depth'])
                loss = self.berhu_loss(preds[valid_mask], depths[valid_mask])

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()

            # 计算指标
            metrics = compute_depth_metrics(preds[valid_mask], depths[valid_mask])
            for k, v in metrics.items():
                metric_sum[k] += v

            if batch_idx % self.config.log_interval == 0 and (not self.is_distributed or self.local_rank == 0):
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch} | Batch {batch_idx:04d} | Loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metric_sum.items()}
        return {'train_loss': avg_train_loss, **avg_metrics}


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        metric_sum = {
            "delta1": 0.0, "delta2": 0.0, "delta3": 0.0,
            "rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0, "silog": 0.0
        }

        for batch_idx, (images, depths) in enumerate(self.val_loader):
            images = images.to(self.device, non_blocking=True)
            depths = depths.to(self.device, non_blocking=True)

            preds = self.model(images)
            preds = F.interpolate(preds, size=depths.shape[-2:], mode='bilinear')

            valid_mask = (depths > 0.01) & (depths < self.config.datasets['nyu']['max_depth'])
            loss = self.berhu_loss(preds[valid_mask], depths[valid_mask])
            total_loss += loss.item()

            # 计算指标
            metrics = compute_depth_metrics(preds[valid_mask], depths[valid_mask])
            for k, v in metrics.items():
                metric_sum[k] += v

            if batch_idx % 50 == 0 and (not self.is_distributed or self.local_rank <= 0):
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Validation | Batch {batch_idx:04d} | Loss: {avg_loss:.4f}")

        avg_val_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in metric_sum.items()}
        return {'val_loss': avg_val_loss, **avg_metrics}

    def run(self):
        best_val_loss = float('inf')
        best_silog = float('inf')
        best_delta1 = 0.0  # 记录最佳 delta1 指标
        log_df = pd.DataFrame()

        if self.dataset == 'nyu':
            log_dir = Path("/workspace/zyw/MobileVim/results/nyu_results")
        elif self.dataset == 'kitti':
            log_dir = Path("/workspace/zyw/MobileVim/results/kitti_results")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "depth_training_log.csv"

        # 初始化日志文件
        if self.local_rank <= 0:
            if not log_path.exists():
                with log_path.open("w") as f:
                    f.write("epoch,train_loss,val_loss,delta1,delta2,delta3,rmse,abs_rel,sq_rel,silog\n")

        for epoch in range(1, self.config.epochs + 1):
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            # 验证
            val_metrics = self.validate()

            # 合并训练与验证指标
            log_entry = {
                'epoch': epoch,
                'train_loss': train_metrics,
                **val_metrics
            }

            # 打印信息
            if self.local_rank <= 0:
                print(f"Epoch {epoch} Summary | "
                      f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"Δ1: {val_metrics['delta1']:.4f}, "
                      f"RMSE: {val_metrics['rmse']:.4f}, "
                      f"Silog: {val_metrics['silog']:.4f}")

                # 写入CSV
                log_line = (f"{epoch},{train_metrics['train_loss']:.4f},{val_metrics['val_loss']:.4f},"
                            f"{val_metrics['delta1']:.4f},{val_metrics['delta2']:.4f},{val_metrics['delta3']:.4f},"
                            f"{val_metrics['rmse']:.4f},{val_metrics['abs_rel']:.4f},{val_metrics['sq_rel']:.4f},"
                            f"{val_metrics['silog']:.4f}\n")

                with log_path.open("a") as f:
                    f.write(log_line)

            # 仅当 epoch >= 10 并且 delta1 是当前最优时才保存模型
            if epoch >= 10 and val_metrics['delta1'] > best_delta1 and (
                    not self.is_distributed or self.local_rank == 0):
                best_delta1 = val_metrics['delta1']
                model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
                save_dir = log_dir
                save_path = save_dir / f"depth_model_delta1_{best_delta1:.4f}.pth"

                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'backbone': self.backbone.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_delta1': best_delta1
                }, save_path)

            # if val_metrics['val_loss'] < best_val_loss and (not self.is_distributed or self.local_rank == 0):
            #     best_val_loss = val_metrics['val_loss']
            #     model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            #     save_path = Path("/workspace/zyw/MobileVim/results/nyu_results") / f"depth_model_val_loss_{best_val_loss:.4f}.pth"
            #     save_path.parent.mkdir(parents=True, exist_ok=True)
            #     torch.save({
            #         'backbone': self.backbone.state_dict(),
            #         'decoder': self.decoder.state_dict(),
            #         'optimizer': self.optimizer.state_dict(),
            #         'epoch': epoch,
            #         'best_loss': best_val_loss
            #     }, save_path)

        if self.local_rank <= 0:
            print(f"Depth training completed with best val loss: {best_val_loss:.4f}, "
                  f"best silog: {best_silog:.4f}, best delta1: {best_delta1:.4f}")

        return best_val_loss


def compute_depth_metrics(pred, gt):
    # 去掉无效区域（如背景或超出最大深度）
    valid_mask = (gt > 0.01) & (gt < 10.0)
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    # 计算 ratio 并限制范围
    ratio = torch.clamp(pred_valid / (gt_valid + 1e-6), 0.1, 10)

    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    diff = gt_valid - pred_valid
    rmse = torch.sqrt((diff ** 2).mean()).item()
    abs_rel = (torch.abs(diff) / (gt_valid + 1e-6)).mean().item()
    sq_rel = ((diff ** 2) / (gt_valid + 1e-6)).mean().item()

    log_diff = torch.log(pred_valid + 1e-6) - torch.log(gt_valid + 1e-6)
    silog = (log_diff ** 2).mean() - (log_diff.mean()) ** 2
    silog = torch.sqrt(silog + 1e-6).item()

    return {
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "silog": silog
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pretrain", "depth"], required=True)
    parser.add_argument("--dataset", choices=["nyu", "kitti"], default="nyu")
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1))
    args = parser.parse_args()

    local_rank = args.local_rank
    if local_rank >= 0:
        local_rank = setup_distributed()

    if args.phase == "pretrain":
        config = ImageNetConfig()
        trainer = ImageNetTrainer(config, local_rank)
        best_acc = trainer.run()
        if local_rank <= 0:
            print(f"Pretraining completed with best accuracy: {best_acc:.2f}%")
    elif args.phase == "depth":
        depth_config = DepthConfig()
        trainer = DepthTrainer(
            config=depth_config,
            pretrained_path="/workspace/zyw/MobileVim/weights/mobilevim_imagenet_top1_0.745.pth",
            local_rank=local_rank,
            dataset = args.dataset
        )
        best_loss = trainer.run()
        if local_rank <= 0:
            print(f"Depth training completed with best loss: {best_loss:.4f}")