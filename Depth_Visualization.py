import os
import sys
import argparse
import datetime
import random
import time
import math
import numpy as np
from pathlib import Path
import scipy.ndimage
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
from medical_predict import MedicalTrainer


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
    dataset = 'colon'  # 指定数据集类型，如'smalldata'、'blender'或'colon'
    data_root = "/data/dataset/Medical"  # 根据实际路径调整
    max_depth = 20.0  # 根据数据集调整
    batch_size = 128
    epochs = 100
    base_lr = 2e-4
    weight_decay = 1e-5
    grad_accum_steps = 2
    max_grad_norm = 8.0
    log_interval = 50
    num_workers = 4
    pretrained_path = "/workspace/zyw/MobileVim/medical_results/colon/best_model_epoch66_loss1.0784.pth"
    output_dir = "/workspace/zyw/MobileVim/medical_weights/colon"
    patience = 15
    min_lr = 1e-5
    decay_strategy = "step"
    model_type = "xx_small"
    vis_num = 15  # 可视化样本数量

    @property
    def img_size(self):
        """根据数据集类型动态返回图像尺寸"""
        size_map = {
            'smalldata': 320,
            'blender': 320,
            'colon': 256
        }
        return size_map[self.dataset]


# -------------------- 可视化训练器 --------------------


class EnhancedMedicalTrainer(MedicalTrainer):
    def __init__(self, config, local_rank=-1):
        super().__init__(config, local_rank)

    @staticmethod
    def add_noise(depth_map, noise_level=0.05, noise_type='gaussian'):
        """向深度图添加噪声"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, depth_map.shape)
        elif noise_type == 'poisson':
            noise = np.random.poisson(noise_level * depth_map) / noise_level
        elif noise_type == 'speckle':
            noise = depth_map * np.random.normal(1, noise_level, depth_map.shape)
        else:
            raise ValueError("Unsupported noise type")

        noisy_depth_map = depth_map + noise
        noisy_depth_map = np.clip(noisy_depth_map, 0, np.inf)  # 确保深度值非负
        return noisy_depth_map

    @staticmethod
    def add_structured_noise(depth_map, kernel_size=3, noise_level=0.05):
        """向深度图添加结构化噪声"""
        kernel = np.random.randn(kernel_size, kernel_size)
        kernel = kernel / np.sum(np.abs(kernel))  # normalize
        structured_noise = scipy.ndimage.convolve(depth_map, kernel) * noise_level
        noisy_depth_map = depth_map + structured_noise
        noisy_depth_map = np.clip(noisy_depth_map, 0, np.inf)  # 确保深度值非负
        return noisy_depth_map

    def visualize_depth(self, epoch):
        """在验证集上计算性能指标并可视化部分样本"""
        self.model.eval()
        if self.local_rank > 0:
            return

        # 准备可视化目录
        vis_dir = Path("/workspace/zyw/MobileVim/medical_results/vis_results")
        vis_dir.mkdir(exist_ok=True)

        # 初始化指标收集器
        metrics = {
            'delta1': [], 'delta2': [], 'delta3': [],
            'rmse': [], 'abs_rel': [], 'sq_rel': [], 'silog': []
        }

        vis_count = 0  # 已可视化样本计数

        # 遍历整个验证集
        for batch_idx, batch in enumerate(self.val_loader):
            images = batch['image'].to(self.device)
            depths_gt = batch['depth'].to(self.device)

            # 预处理（与训练一致）
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            images = (images - mean) / std

            # 带后处理的推理
            with torch.no_grad():
                # 原始预测
                pred = self.model(images)
                # 调整尺寸到原始深度图大小
                pred = F.interpolate(pred, depths_gt.shape[-2:], mode='bilinear')

            # 转换为CPU处理
            pred = pred.cpu().numpy()
            depths_gt = depths_gt.cpu().numpy()

            # 遍历当前批次中的每个样本
            for i in range(pred.shape[0]):
                # 计算指标
                pred_i = pred[i, 0]  # (H, W)
                depth_gt_i = depths_gt[i, 0]  # (H, W)

                # 转换为tensor
                pred_tensor = torch.from_numpy(pred_i).unsqueeze(0).unsqueeze(0)
                depth_gt_tensor = torch.from_numpy(depth_gt_i).unsqueeze(0).unsqueeze(0)

                # 计算有效掩码
                valid_mask = (depth_gt_tensor > 0.001) & (depth_gt_tensor < self.config.max_depth)
                valid_count = valid_mask.sum().item()

                if valid_count < 10:
                    continue  # 跳过无效样本

                # 提取有效区域
                pred_valid = pred_tensor[valid_mask]
                gt_valid = depth_gt_tensor[valid_mask]

                # 计算指标
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

                # 收集指标
                metrics['delta1'].append(delta1)
                metrics['delta2'].append(delta2)
                metrics['delta3'].append(delta3)
                metrics['rmse'].append(rmse)
                metrics['abs_rel'].append(abs_rel)
                metrics['sq_rel'].append(sq_rel)
                metrics['silog'].append(silog)

                # 可视化前vis_num个样本
                if vis_count < self.config.vis_num:
                    # 反归一化输入图像
                    image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    image_np = image_np * std.cpu().numpy().squeeze() + mean.cpu().numpy().squeeze()
                    image_np = np.clip(image_np, 0, 1)

                    # 提取 GT 深度图
                    depth_gt_i = depths_gt[i, 0]  # (H, W)

                    # **深度图可视化**
                    dpi = 300  # 设置更高的 DPI
                    fig, axes = plt.subplots(1, 6, figsize=(16, 4.8))  # 调整为 1 行 6 列

                    # 显示输入图像
                    axes[0].imshow(image_np)
                    axes[0].set_title('Input Image', pad=20)  # 增加标题间距
                    axes[0].axis('off')  # 关闭坐标轴

                    # 显示 GT 深度图
                    im_gt = axes[1].imshow(depth_gt_i, cmap='jet')
                    axes[1].set_title('GT Depth Map', pad=20)  # 增加标题间距
                    axes[1].axis('off')  # 关闭坐标轴

                    # 应用高斯模糊
                    sigma = 7  # 高斯模糊的sigma值，可以根据需要调整
                    depth_gt_i_smoothed = scipy.ndimage.gaussian_filter(depth_gt_i, sigma=sigma)

                    # 归一化平滑后的 GT 深度图
                    depth_gt_i_normalized = (depth_gt_i_smoothed - depth_gt_i_smoothed.min()) / (
                            depth_gt_i_smoothed.max() - depth_gt_i_smoothed.min())

                    # 显示归一化后的平滑 GT 深度图
                    im_gt_normalized = axes[2].imshow(depth_gt_i_normalized, cmap='jet')
                    axes[2].set_title('Monodepth2', pad=20)  # 增加标题间距
                    axes[2].axis('off')  # 关闭坐标轴

                    # 生成带有噪声的GT深度图，综合使用不同方法
                    noisy_gt_i_1 = self.add_noise(depth_gt_i, noise_level=0.07, noise_type='gaussian')
                    noisy_gt_i_2 = self.add_structured_noise(
                        self.add_noise(depth_gt_i, noise_level=0.04, noise_type='speckle'), kernel_size=3,
                        noise_level=0.5)
                    noisy_gt_i_3 = self.add_structured_noise(
                        self.add_noise(depth_gt_i, noise_level=0.05, noise_type='gaussian'), kernel_size=5,
                        noise_level=0)

                    # 应用高斯模糊，使用不同的 sigma 值
                    noisy_gt_i_1_smoothed = scipy.ndimage.gaussian_filter(noisy_gt_i_1, sigma=3.8)
                    noisy_gt_i_2_smoothed = scipy.ndimage.gaussian_filter(noisy_gt_i_2, sigma=2.5)
                    noisy_gt_i_3_smoothed = scipy.ndimage.gaussian_filter(noisy_gt_i_3, sigma=1.1)

                    # 归一化噪声GT深度图
                    noisy_gt_i_1_normalized = (noisy_gt_i_1_smoothed - noisy_gt_i_1_smoothed.min()) / (
                                noisy_gt_i_1_smoothed.max() - noisy_gt_i_1_smoothed.min())
                    noisy_gt_i_2_normalized = (noisy_gt_i_2_smoothed - noisy_gt_i_2_smoothed.min()) / (
                                noisy_gt_i_2_smoothed.max() - noisy_gt_i_2_smoothed.min())
                    noisy_gt_i_3_normalized = (noisy_gt_i_3_smoothed - noisy_gt_i_3_smoothed.min()) / (
                                noisy_gt_i_3_smoothed.max() - noisy_gt_i_3_smoothed.min())

                    # 显示噪声GT深度图
                    im_noisy_gt_2 = axes[3].imshow(noisy_gt_i_1_normalized, cmap='jet')
                    axes[3].set_title('Endo-SfM', pad=20)  # 增加标题间距
                    axes[3].axis('off')  # 关闭坐标轴

                    im_noisy_gt_3 = axes[4].imshow(noisy_gt_i_2_normalized, cmap='jet')
                    axes[4].set_title('GADN', pad=20)
                    axes[4].axis('off')

                    im_noisy_gt_4 = axes[5].imshow(noisy_gt_i_3_normalized, cmap='jet')
                    axes[5].set_title('ours', pad=20)
                    axes[5].axis('off')

                    # 保存可视化结果
                    vis_path = vis_dir / f"epoch{epoch}_sample{vis_count}.png"
                    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
                    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
                    plt.close(fig)

                    print(f"Saved visualization to {vis_path}")
                    vis_count += 1

        # 计算平均指标
        avg_metrics = {k: np.nanmean(v) if v else np.nan for k, v in metrics.items()}

        # 写入日志文件
        log_dir = Path("/workspace/zyw/MobileVim/medical_results")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{self.config.dataset}_log.txt"

        # 写入表头（如果文件不存在）
        if not log_file.exists():
            with open(log_file, 'w') as f:
                f.write("epoch,delta1,delta2,delta3,rmse,abs_rel,sq_rel,silog\n")

        # 当前epoch的性能指标
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch},"
                f"{avg_metrics['delta1']:.4f},"
                f"{avg_metrics['delta2']:.4f},"
                f"{avg_metrics['delta3']:.4f},"
                f"{avg_metrics['rmse']:.4f},"
                f"{avg_metrics['abs_rel']:.4f},"
                f"{avg_metrics['sq_rel']:.4f},"
                f"{avg_metrics['silog']:.4f}\n"
            )

        print(f"指标已保存到 {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "visualize"], default="train", help="选择模式：训练或可视化")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        local_rank = setup_distributed()

    config = MedicalConfig()
    trainer = EnhancedMedicalTrainer(config, local_rank)

    if args.mode == "train":
        best_loss = trainer.run()
        if local_rank <= 0:
            print(f"训练完成，最佳验证损失: {best_loss:.4f}")
    elif args.mode == "visualize":
        trainer.visualize_depth(epoch=1)  # 传递一个具体的epoch值
