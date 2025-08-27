from functools import partial

import pywt
import pywt.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.layers import DropPath, SqueezeExcite
from timm.models.vision_transformer import trunc_normal_
from typing import List, Tuple, Optional

from model import MODEL
from model.lib_mamba.vmambanew import SS2D
from model.defattn_decoder import MSDeformAttnPixelDecoder
from model.id_module import AFP, ISD


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """创建小波变换的分解和重构滤波器组

    Args:
        wave (str): 小波名称，如'db1'
        in_size (int): 输入数据的通道数（分解滤波器的输入通道数）
        out_size (int): 输出数据的通道数（重构滤波器的输出通道数）
        type (torch.dtype): 张量数据类型，默认为torch.float

    Returns:
        dec_filters (Tensor): 分解滤波器组，形状为 [in_size*4, 1, H, W]
        rec_filters (Tensor): 重构滤波器组，形状为 [out_size*4, 1, H, W]
    """
    # 初始化小波对象，获取分解和重构滤波器系数
    w = pywt.Wavelet(wave)  # 创建小波对象

    # 分解滤波器处理（用于小波分解）
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # 获取分解高通滤波器并反转顺序
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  # 获取分解低通滤波器并反转顺序

    # 通过外积生成四个二维分解滤波器 (LL, LH, HL, HH)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # 低低组合 (LL)
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # 低高组合 (LH)
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # 高低组合 (HL)
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 高高组合 (HH)
    ], dim=0)  # 输出形状 [4, H, W]，H/W为滤波器长度

    # 扩展分解滤波器维度以匹配输入通道数
    dec_filters = dec_filters[:, None]  # 增加通道维度 → [4, 1, H, W]
    dec_filters = dec_filters.repeat(in_size, 1, 1, 1)  # 按输入通道数重复 → [in_size*4, 1, H, W]

    # 重构滤波器处理（用于小波重构）
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])  # 反转+翻转顺序
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])  # 确保与分解相位匹配

    # 通过外积生成四个二维重构滤波器 (LL, LH, HL, HH)
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),  # 低低组合 (LL)
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  # 低高组合 (LH)
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),  # 高低组合 (HL)
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)  # 高高组合 (HH)
    ], dim=0)  # 输出形状 [4, H, W]

    # 扩展重构滤波器维度以匹配输出通道数
    rec_filters = rec_filters[:, None]  # 增加通道维度 → [4, 1, H, W]
    rec_filters = rec_filters.repeat(out_size, 1, 1, 1)  # 按输出通道数重复 → [out_size*4, 1, H, W]

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    """应用二维小波变换，将输入信号分解为四个子带

    Args:
        x (Tensor): 输入张量，形状为 [batch_size, channels, height, width]
        filters (Tensor): 分解滤波器组，形状为 [channels*4, 1, H, W]

    Returns:
        Tensor: 小波变换结果，形状为 [batch_size, channels, 4, height//2, width//2]
            其中第三个维度对应四个子带：LL(低低), LH(低高), HL(高低), HH(高高)
    """
    # 获取输入张量的维度信息
    b, c, h, w = x.shape  # batch大小, 通道数, 高度, 宽度

    # 计算填充大小：确保卷积后尺寸正确减半
    # 滤波器尺寸为 (H, W)，填充量为滤波器半长减1（适配对称小波滤波器）
    pad = (filters.shape[2] // 2 - 1,  # 高度方向填充量
           filters.shape[3] // 2 - 1)  # 宽度方向填充量

    # 执行分组卷积：每个输入通道独立应用四个滤波器
    # 输出形状 [b, c*4, h//2, w//2]（stride=2实现下采样）
    x = F.conv2d(x,
                 filters,
                 stride=2,  # 步长2实现空间尺寸减半
                 groups=c,  # 分组数=通道数，实现通道独立处理
                 padding=pad)  # 对称填充保持相位对齐

    # 重塑张量结构：将四个子带组织到独立维度
    # 从 [b, c*4, h', w'] → [b, c, 4, h', w'] 其中 h' = h//2, w' = w//2
    x = x.reshape(b, c, 4, h // 2, w // 2)

    return x


def inverse_wavelet_transform(x, filters):
    """应用二维逆小波变换，将四个子带重构成原始尺寸信号

    Args:
        x (Tensor): 输入张量，形状为 [batch_size, channels, 4, h//2, w//2]
            第三个维度包含四个子带：LL(低低), LH(低高), HL(高低), HH(高高)
        filters (Tensor): 重构滤波器组，形状为 [channels*4, 1, H, W]

    Returns:
        Tensor: 重构后的信号，形状为 [batch_size, channels, height, width]
            其中 height = h//2 * 2, width = w//2 * 2 恢复原始空间尺寸
    """
    # 获取输入张量的维度信息（_占位符对应子带维度4）
    b, c, _, h_half, w_half = x.shape  # batch大小, 通道数, 子带数, 半高, 半宽

    # 计算填充大小：与正变换对称，确保卷积转置后尺寸正确加倍
    pad = (filters.shape[2] // 2 - 1,  # 高度方向填充量
           filters.shape[3] // 2 - 1)  # 宽度方向填充量

    # 合并子带维度到通道维度：将四个子带展开为独立通道
    # [b, c, 4, h', w'] → [b, c*4, h', w'] (h'=h_half, w'=w_half)
    x = x.reshape(b, c * 4, h_half, w_half)

    # 执行分组转置卷积：每个通道组独立重构信号
    # 输出形状 [b, c, h, w] 其中 h = h_half*2, w = w_half*2
    x = F.conv_transpose2d(x,
                           filters,
                           stride=2,  # 步长2实现空间尺寸加倍
                           groups=c,  # 分组数=通道数，保持通道独立处理
                           padding=pad)  # 对称填充确保相位对齐

    return x


class MBWTConv2d(nn.Module):
    """多分支小波变换卷积模块（Multi-Branch Wavelet Transform Convolution）

    特点：
    - 结合小波多分辨率分析与深度卷积
    - 支持多级小波分解与重构
    - 集成状态空间模型（SSM）进行全局特征调制

    主要流程：
    输入 → 多级小波分解 → 高频分量卷积处理 → 逆小波重构 → 全局注意力融合 → 输出

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数（需等于in_channels）
        kernel_size (int): 高频分支卷积核尺寸，默认为5
        stride (int): 空间下采样步长，默认为1
        bias (bool): 是否使用偏置项，默认为True
        wt_levels (int): 小波分解级数，默认为1
        wt_type (str): 小波基类型，如'db1'，默认为'db1'
        ssm_ratio (int): 状态空间模型扩展比，默认为1
        forward_type (str): SSM前向计算版本，默认为"v05"
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True,
                 wt_levels=1, wt_type='db1', ssm_ratio=1, forward_type="v05"):
        super(MBWTConv2d, self).__init__()
        assert in_channels == out_channels  # 当前设计要求输入输出通道相同

        # 基础参数设置
        self.in_channels = in_channels
        self.wt_levels = wt_levels  # 小波分解深度（默认为1级）
        self.stride = stride  # 空间下采样步长
        self.dilation = 1  # 扩张率（当前未使用）

        # 小波滤波器初始化（不可训练参数）
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 绑定小波变换函数（通过偏函数预设滤波器参数）
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)  # 分解函数
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)  # 重构函数

        # 全局特征调制模块
        self.global_atten = SS2D(
            d_model=in_channels,  # 特征维度
            d_state=1,  # 状态维度
            ssm_ratio=ssm_ratio,  # SSM隐藏层扩展比例
            initialize="v2",  # 参数初始化方式
            forward_type=forward_type,  # 前向计算模式
            channel_first=True,  # 通道维度在前
            k_group=2  # 分组卷积数
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])  # 基础缩放模块

        # 构建多级小波处理分支
        self.wavelet_convs = nn.ModuleList([
            # 深度可分离卷积处理高频成分（每组处理4个通道：LH/HL/HH）
            nn.Conv2d(
                in_channels * 4,  # 输入通道（4子带 × 原通道）
                in_channels * 4,  # 输出通道（保持相同）
                kernel_size,
                padding='same',  # 保持空间尺寸
                stride=1,
                dilation=1,
                groups=in_channels * 4,  # 深度可分离分组（每组处理单个子带）
                bias=False  # 无偏置项
            ) for _ in range(self.wt_levels)
        ])

        # 每级小波分支的缩放系数（初始化为0.1，促进稳定训练）
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        # 下采样处理（当stride>1时）
        if self.stride > 1:
            # 使用逐通道1x1卷积实现步长下采样（不可训练参数）
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(
                x_in, self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=in_channels
            )
        else:
            self.do_stride = None

    def forward(self, x):
        """前向传播过程

        处理流程：
        1. 多级小波分解：逐级提取低频分量，存储高频分量
        2. 高频分量处理：对每级的LH/HL/HH进行深度卷积
        3. 逆小波重构：从最深层级开始逐级融合高低频信息
        4. 全局特征融合：通过SSM增强特征，并与小波分支结果相加
        5. 下采样输出：根据stride参数决定是否进行空间下采样
        """
        # 初始化各层级存储容器
        x_ll_in_levels = []  # 存储每级的低频分量（LL）
        x_h_in_levels = []  # 存储每级的高频分量（LH/HL/HH）
        shapes_in_levels = []  # 记录每级输入尺寸（用于逆变换时裁剪）

        # 初始低频分量为输入本身
        curr_x_ll = x

        # 前向分解阶段：多级小波分解 -------------------------------------------
        for i in range(self.wt_levels):
            # 记录当前层级输入尺寸
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 填充处理（当尺寸为奇数时补零）
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)  # 右下方补零

            # 执行小波分解，得到4个子带 [LL, LH, HL, HH]
            curr_x = self.wt_function(curr_x_ll)  # 输出形状 [B, C, 4, H//2, W//2]

            # 分离低频分量（LL）供下一级分解使用
            curr_x_ll = curr_x[:, :, 0, :, :]  # 索引0对应LL子带

            # 处理高频分量（LH/HL/HH）
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  # 合并子带维度到通道
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))  # 深度卷积+缩放
            curr_x_tag = curr_x_tag.reshape(shape_x)  # 恢复子带维度

            # 存储处理后的高低频信息
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # 存储当前级的处理后LL
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # 存储当前级的处理后高频（LH/HL/HH）

        # 逆向重构阶段：从最深层级开始融合 -------------------------------------
        if self.wt_levels == 0:
            # 当不进行小波分解时，直接返回全局特征调制结果
            x = self.base_scale(self.global_atten(x))
            if self.do_stride is not None:
                x = self.do_stride(x)
            return x  # 提前返回避免后续空循环

        next_x_ll = torch.zeros_like(curr_x_ll, device=x.device)  # 初始化最深层的低频残差

        for i in range(self.wt_levels - 1, -1, -1):  # 倒序处理层级（从最深到最浅）
            # 取出当前层级存储的信息
            curr_x_ll = x_ll_in_levels.pop()  # 当前级处理后的LL
            curr_x_h = x_h_in_levels.pop()  # 当前级处理后的高频
            curr_shape = shapes_in_levels.pop()  # 原始输入尺寸

            # 低频融合：当前级LL + 深层传递的残差
            curr_x_ll = curr_x_ll + next_x_ll

            # 拼接高低频信息（LL与LH/HL/HH）
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)  # 恢复子带维度

            # 执行逆小波变换，重构上一级低频分量
            next_x_ll = self.iwt_function(curr_x)  # 输出形状 [B, C, H, W]

            # 裁剪尺寸（处理可能的填充）
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 全局特征融合 --------------------------------------------------
        x_tag = next_x_ll  # 最终重构结果
        assert len(x_ll_in_levels) == 0  # 确认所有层级已处理

        # 原输入通过SSM模块进行全局特征调制
        x = self.base_scale(self.global_atten(x))
        # 残差连接：全局特征 + 小波分支结果
        x = x + x_tag

        # 下采样输出（如果需要）
        if self.do_stride is not None:
            x = self.do_stride(x)  # 执行步长下采样

        return x


class _ScaleModule(nn.Module):
    """可学习的缩放模块，实现逐元素或通道级特征缩放

    功能描述：
    - 对输入张量进行可学习的缩放操作，每个缩放因子对应输入的一个维度
    - 常用于调整不同特征通道的重要性（类似注意力机制的简化版本）
    - 初始化阶段通过init_scale控制初始缩放幅度

    Args:
        dims (list/tuple): 缩放权重的维度，例如：
            - [1, C, 1, 1] 实现逐通道缩放（常用）
            - [1] 实现全局标量缩放
        init_scale (float): 缩放因子的初始值，默认为1.0（等同单位变换）
        init_bias (float): 偏置项初始值（当前版本未实现，保留接口）
    """

    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims  # 权重张量的形状

        # 初始化可学习缩放参数（广播机制适配输入维度）
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)  # 创建形状为dims的全1张量，乘以初始缩放值

        # 当前版本未实现偏置项（保留参数接口供后续扩展）
        self.bias = None  # 可扩展为：nn.Parameter(torch.zeros(*dims) + init_bias)

    def forward(self, x):
        """前向传播：执行元素级缩放

        输入输出形状：
        - 输入 x: [B, C, H, W] 或其它任意形状（需与self.dims广播兼容）
        - 输出:  与输入x相同形状，每个元素乘以对应位置的缩放因子

        广播机制示例：
        - 当dims=[1, C, 1, 1]时，对4D输入执行逐通道缩放
        - 当dims=[1]时，对所有元素执行相同缩放
        """
        return torch.mul(self.weight, x)  # 元素级乘法（支持广播）


class DWConv2d_BN_ReLU(nn.Sequential):
    """深度可分离卷积块（含BN与ReLU），支持卷积层与BN层的参数融合

    结构说明：
    [3x3深度卷积 → BN → ReLU → 1x1深度卷积 → BN]
    特点：
    - 通过两次深度卷积实现空间+通道特征提取
    - 所有卷积层均为深度可分离（分组数=输入通道数）
    - 提供fuse方法合并卷积与BN层，提升推理速度

    使用注意：
    - out_channels必须是in_channels的整数倍（因groups=in_channels）
    - 适用于轻量化网络设计，参数量远小于标准卷积

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数（需为in_channels的整数倍）
        kernel_size (int): 空间卷积核尺寸，默认为3
        bn_weight_init (float): BN层权重的初始值，默认为1
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn_weight_init=1):
        super().__init__()
        # 3x3深度卷积（逐通道空间特征提取）
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=kernel_size // 2,
                                  groups=in_channels,
                                  bias=False))
        # 批归一化与激活
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        # 1x1深度卷积（跨通道特征融合）
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  groups=1,  # 保持深度可分离特性
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # 初始化BN参数（稳定训练）
        nn.init.constant_(self.bn1.weight, bn_weight_init)  # γ初始值
        nn.init.constant_(self.bn1.bias, 0)  # β初始值
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """融合卷积层与BN层参数，生成推理优化结构

        融合公式：
        w_fused = w_conv * (γ / sqrt(σ^2 + ε))
        b_fused = β - (γ * μ) / sqrt(σ^2 + ε)

        返回：
            nn.Sequential: 包含融合后卷积层的新序列
        """
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        # 融合3x3卷积与BN1 ----------------------------------------------
        # 计算融合后的权重
        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]  # 广播乘法

        # 计算融合后的偏置
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        # 创建融合后的3x3卷积层
        fused_dwconv3x3 = nn.Conv2d(
            w1.size(1) * dwconv3x3.groups,  # 输入通道 = 单组输入通道数 × 组数
            w1.size(0),  # 输出通道数保持不变
            w1.shape[2:],  # 卷积核尺寸
            stride=dwconv3x3.stride,
            padding=dwconv3x3.padding,
            dilation=dwconv3x3.dilation,
            groups=dwconv3x3.groups,
            device=dwconv3x3.weight.device
        )
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # 融合1x1卷积与BN2 ----------------------------------------------
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(
            w2.size(1) * dwconv1x1.groups,
            w2.size(0),
            w2.shape[2:],
            stride=dwconv1x1.stride,
            padding=dwconv1x1.padding,
            dilation=dwconv1x1.dilation,
            groups=dwconv1x1.groups,
            device=dwconv1x1.weight.device
        )
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # 构建优化后的序列模型
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model


class PyramidDWConv(nn.Module):
    """金字塔深度可分离卷积模块
    结构：
       输入 → 通道均分3分支 → [3x3/5x5/7x7深度卷积] → 特征拼接 → 1x1卷积融合

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_sizes (list): 各分支卷积核尺寸，默认为[3,5,7]
        bn_weight_init (float): BN层权重初始化值，默认为1.0
    """

    def __init__(self, in_channels, out_channels,
                 kernel_sizes=[3, 5, 7], bn_weight_init=1.0):
        super().__init__()

        # 将输入通道均分到三个分支（处理无法整除的情况）
        self.split_sizes = self._split_channels(in_channels, 3)

        # 确保总通道数等于输入通道数
        assert sum(self.split_sizes) == in_channels, "总通道数不等于输入通道数"

        # 调试信息
        # print(f"PyramidDWConv: in_channels={in_channels}, split_sizes={self.split_sizes}")

        # 构建三个不同尺度的深度卷积分支
        self.branches = nn.ModuleList()
        for i, ksize in enumerate(kernel_sizes):
            branch = nn.Sequential(
                # 深度可分离卷积（分组数=输入通道数）
                nn.Conv2d(
                    self.split_sizes[i],  # 单分支输入通道
                    self.split_sizes[i],  # 保持通道数
                    kernel_size=ksize,
                    padding=ksize // 2,  # 保持空间尺寸
                    groups=self.split_sizes[i],  # 深度卷积
                    bias=False
                ),
                nn.BatchNorm2d(self.split_sizes[i]),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
            # 初始化BN参数
            nn.init.constant_(branch[1].weight, bn_weight_init)
            nn.init.constant_(branch[1].bias, 0)

        # 特征融合层（1x1卷积调整通道）
        self.fusion = Conv2d_BN(
            in_channels,  # 输入总通道（三分支concat后）
            out_channels,
            ks=1,
            bn_weight_init=bn_weight_init
        )

    def _split_channels(self, total, num_groups):
        """均分通道数，处理无法整除的情况"""
        base = total // num_groups
        remainder = total % num_groups
        split = [base + 1 if i < remainder else base for i in range(num_groups)]
        return split

    def forward(self, x):
        # 拆分输入到三个分支
        x_split = torch.split(x, self.split_sizes, dim=1)

        # 并行处理各分支
        branch_outs = []
        for i, branch in enumerate(self.branches):
            out = branch(x_split[i])  # [B, C_i, H, W]
            branch_outs.append(out)

        # 通道维度拼接
        x = torch.cat(branch_outs, dim=1)  # [B, C1+C2+C3, H, W]

        # 特征融合
        x = self.fusion(x)  # [B, out_channels, H, W]
        return x



class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        # 添加一个卷积层到Sequential模块中
        # 输入通道数为a，输出通道数为b，卷积核大小为ks，步长为stride，填充为pad，扩张率为dilation，分组数为groups
        # 注意：这里卷积层的bias设置为False，因为后面会接一个BatchNorm层
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))

        # 添加一个BatchNorm层到Sequential模块中
        # 参数为输出通道数b
        self.add_module('bn', torch.nn.BatchNorm2d(b))

        # 初始化BatchNorm层的权重为bn_weight_init
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)

        # 初始化BatchNorm层的偏置为0
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # 获取卷积层和BatchNorm层
        c, bn = self._modules.values()

        # 计算融合后的权重
        # w = bn.weight / sqrt(bn.running_var + bn.eps)
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5

        # 将权重w扩展到与卷积层权重相同的形状
        w = c.weight * w[:, None, None, None]

        # 计算融合后的偏置
        # b = bn.bias - bn.running_mean * bn.weight / sqrt(bn.running_var + bn.eps)
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5

        # 创建一个新的卷积层，参数与原来的卷积层相同
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)

        # 将融合后的权重和偏置复制到新的卷积层中
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)

        # 返回融合后的卷积层
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        # 添加一个BatchNorm1d层，输入维度为a
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        # 添加一个Linear层，输入维度为a，输出维度为b，是否使用偏置由bias参数决定
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # 使用trunc_normal_函数初始化Linear层的权重，标准差为std
        trunc_normal_(self.l.weight, std=std)
        # 如果使用偏置，则将其初始化为0
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # 获取BatchNorm层和Linear层
        bn, l = self._modules.values()
        # 计算融合后的权重：w = bn.weight / sqrt(bn.running_var + bn.eps)
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        # 计算融合后的偏置：b = bn.bias - bn.running_mean * bn.weight / sqrt(bn.running_var + bn.eps)
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        # 将权重w与Linear层的权重相乘
        w = l.weight * w[None, :]
        # 计算最终的偏置
        if l.bias is None:
            b = b @ self.l.weight.T  # 如果Linear层没有偏置，则只计算BatchNorm的影响
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias  # 如果Linear层有偏置，则加上Linear层的偏置
        # 创建一个新的Linear层，输入维度为w.size(1)，输出维度为w.size(0)
        m = torch.nn.Linear(w.size(1), w.size(0))
        # 将融合后的权重和偏置复制到新的Linear层中
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        # 返回融合后的Linear层
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        # 定义隐藏层维度为输入维度的4倍
        hid_dim = int(dim * 4)
        # 第一层卷积：输入维度为dim，输出维度为hid_dim，卷积核大小为1x1，步长为1，填充为0
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        # 激活函数：ReLU
        self.act = torch.nn.ReLU()
        # 第二层卷积：输入维度为hid_dim，输出维度为hid_dim，卷积核大小为3x3，步长为2，填充为1，分组数为hid_dim
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        # Squeeze-and-Excitation模块：输入维度为hid_dim，压缩比例为0.25
        self.se = SqueezeExcite(hid_dim, .25)
        # 第三层卷积：输入维度为hid_dim，输出维度为out_dim，卷积核大小为1x1，步长为1，填充为0
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        # 前向传播：依次通过conv1 -> ReLU -> conv2 -> ReLU -> SE -> conv3
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        # 保存传入的模块m
        self.m = m
        # 保存dropout概率
        self.drop = drop

    def forward(self, x):
        # 如果是训练模式且dropout概率大于0
        if self.training and self.drop > 0:
            # 生成一个随机张量，形状为(x.size(0), 1, 1, 1)，值在[0, 1)之间
            # 判断随机值是否大于dropout概率，大于的保留，小于的丢弃
            # 保留的值除以(1 - self.drop)以保持期望值不变
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            # 如果不是训练模式或dropout概率为0，直接返回x + m(x)
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        # 第一层卷积：输入维度为ed，输出维度为h
        self.pw1 = Conv2d_BN(ed, h)
        # 激活函数：ReLU
        self.act = torch.nn.ReLU()
        # 第二层卷积：输入维度为h，输出维度为ed，BatchNorm的权重初始化为0
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        # 前向传播：依次通过pw1 -> ReLU -> pw2
        x = self.pw2(self.act(self.pw1(x)))
        return x


class FFN_Optimized(torch.nn.Module):
    def __init__(self, ed, expansion=4, groups=4):
        """
        Args:
            ed (int): 输入/输出通道数
            expansion (int): 中间层通道扩展倍数，必须为groups的整数倍
            groups (int): 压缩阶段的分组数（默认4组）
        """
        super().__init__()
        h = ed * expansion

        # 扩展阶段：深度可分离卷积（Groups=输入通道数）
        self.pw1 = Conv2d_BN(ed, h, ks=1, groups=ed)  # 参数量: ed*expansion

        # 压缩阶段：分组卷积（Groups需能整除输入/输出通道）
        self.pw2 = Conv2d_BN(h, ed, ks=1, groups=groups, bn_weight_init=0)  # 参数量: (h*ed)/groups

        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

def nearest_multiple_of_16(n):
    # 如果n已经是16的倍数，直接返回n
    if n % 16 == 0:
        return n
    else:
        # 计算小于n的最大16的倍数
        lower_multiple = (n // 16) * 16
        # 计算大于n的最小16的倍数
        upper_multiple = lower_multiple + 16

        # 返回离n最近的16的倍数
        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple

class MobileViMModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        # 输入维度
        self.dim = dim
        # 计算全局通道数，确保是16的倍数
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        # 计算局部通道数，确保全局通道数 + 局部通道数 <= 总维度
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        # 计算恒等映射通道数
        self.identity_channels = self.dim - self.global_channels - self.local_channels

        # 如果局部通道数不为0，定义局部操作（金字塔深度可分离卷积）
        if self.local_channels != 0:
            self.local_op = PyramidDWConv(in_channels=self.local_channels,
                                          out_channels=self.local_channels,
                                          kernel_sizes=[3, 5, 7]
                                          )
        else:
            self.local_op = nn.Identity()

        # 如果全局通道数不为0，定义全局操作（MBWTConv2d，一种基于SSM的卷积）
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(self.global_channels, self.global_channels, kernels, wt_levels=1, ssm_ratio=ssm_ratio, forward_type=forward_type,)
        else:
            self.global_op = nn.Identity()

        # 定义投影层：ReLU + Conv2d_BN
        self.proj = nn.Sequential(
            DWConv2d_BN_ReLU(dim, dim // 2, bn_weight_init=0),  # 深度卷积降维
            Conv2d_BN(dim // 2, dim, groups=4, bn_weight_init=0)  # 分组卷积恢复维度
        )

    def forward(self, x):  # x (B,C,H,W)

        # 动态分割输入张量，确保每个部分都是张量
        parts = []
        if self.global_channels > 0:
            parts.append(self.global_channels)
        if self.local_channels > 0:
            parts.append(self.local_channels)
        if self.identity_channels > 0:
            parts.append(self.identity_channels)

        # 确保至少有一个部分
        if not parts:
            raise ValueError("All channel counts (global, local, identity) are zero.")

        # 动态分割输入张量
        splits = torch.split(x, parts, dim=1)

        # 初始化 x1, x2, x3，默认为零张量
        if self.global_channels > 0:
            x1 = splits[0]
        else:
            x1 = torch.zeros_like(x[:, :self.global_channels])

        if self.local_channels > 0:
            x2 = splits[1]
        else:
            x2 = torch.zeros_like(x[:, :self.local_channels])

        if self.identity_channels > 0:
            x3 = splits[2]
        else:
            x3 = torch.zeros_like(x[:, :self.identity_channels])


        # 检查 x1, x2, x3 是否为张量
        assert isinstance(x1, torch.Tensor), f"x1 is not a tensor: {type(x1)}"
        assert isinstance(x2, torch.Tensor), f"x2 is not a tensor: {type(x2)}"
        assert isinstance(x3, torch.Tensor), f"x3 is not a tensor: {type(x3)}"

        # 对全局部分应用全局操作
        if self.global_channels > 0:
            x1 = self.global_op(x1)
        else:
            print("No global operation applied, x1 remains unchanged.")

        # 对局部部分应用局部操作
        if self.local_channels > 0:
            x2 = self.local_op(x2)
        else:
            print("No local operation applied, x2 remains unchanged.")

        # 将处理后的全局部分、局部部分和恒等映射部分拼接，并通过投影层
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x




class MobileViMBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        # 输入维度
        self.dim = dim
        # 定义注意力模块（MobileMambaModule）
        self.attn = MobileViMModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                           kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type,)

    def forward(self, x):
        # 将输入x通过注意力模块
        x = self.attn(x)
        return x


class MobileViMBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        # 第一个深度可分离卷积 + BN，并包装为残差模块
        self.PyDWConv0 = Residual(PyramidDWConv(ed, ed, kernel_sizes=[3,5,7], bn_weight_init=0.))
        # 第一个前馈网络，并包装为残差模块
        # self.ffn0 = Residual(FFN(ed, int(ed * 2)))
        self.ffn0 = Residual(FFN_Optimized(ed, expansion=2, groups=4))

        # 根据类型选择是否添加MobileMambaBlockWindow模块
        if type == 's':
            self.mixer = Residual(MobileViMBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio,
                                                       kernels=kernels, ssm_ratio=ssm_ratio,forward_type=forward_type))

        # 第二个深度可分离卷积 + BN，并包装为残差模块
        self.PyDWConv1 = Residual(PyramidDWConv(ed, ed, kernel_sizes=[3,5,7], bn_weight_init=0.))
        # 第二个前馈网络，并包装为残差模块
        # self.ffn1 = Residual(FFN(ed, int(ed * 2)))
        self.ffn1 = Residual(FFN_Optimized(ed, expansion=2, groups=4))

        # 是否使用跳跃连接
        self.has_skip = has_skip
        # DropPath模块，用于随机丢弃路径
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        # 保存输入作为跳跃连接的捷径
        shortcut = x
        # 依次通过dw0 -> ffn0 -> mixer -> dw1 -> ffn1
        x = self.ffn1(self.PyDWConv1(self.mixer(self.ffn0(self.PyDWConv0(x)))))
        # 如果使用跳跃连接，则将捷径与drop_path后的输出相加
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x



class MobileViM(torch.nn.Module):
    def __init__(
            self,
            model_type='xx_small',
            num_classes=1000,
            in_chans=3,
            img_size=224,
            embed_dims=[192, 384, 448],
            initial_channels=192,
            # 编码器参数
            global_ratio=None,
            local_ratio=None,
            kernels=None,
            drop_path=0.,
            ssm_ratio=1,
            forward_type="v052d",
            # 解码器参数
            decoder_hidden_dim=256,
            decoder_depths=3,
            decoder_heads=4,
            decoder_anchor_points=4,
            decoder_expansion=2,
            afp_latent_dim=128,
            afp_num_latents=64,
            afp_depths=2,
            isd_depths=2
    ):
        super().__init__()

        # --- 编码器部分 ---
        self.embed_dims = embed_dims

        if global_ratio is None:
            global_ratio = [0.8, 0.7, 0.6]
        if local_ratio is None:
            local_ratio = [0.2, 0.2, 0.3]
        if kernels is None:
            kernels = [7, 5, 3]

        assert model_type in ['small', 'x_small', 'xx_small']
        self.model_type = model_type
        depth_config = {
            'small': [3, 4, 5],
            'x_small': [2, 3, 3],
            'xx_small': [1, 2, 2]
        }
        depth = depth_config[model_type]

        # 参数校验
        assert len(embed_dims) == 3, "需要3个阶段的embed_dims"
        assert img_size % 32 == 0, "输入尺寸必须是32的倍数"

        # Patch embedding层：将输入图像转换为特征图
        self.patch_embed = torch.nn.Sequential(
            # 初始下采样：普通卷积（步长2）
            Conv2d_BN(
                a=in_chans,
                b=initial_channels // 8,
                ks=3,
                stride=2,
                pad=1,
                groups=1
            ),
            torch.nn.ReLU(),
            # 四个DW块：第二个步长2，其余步长1
            DWConv2d_BN_ReLU(initial_channels // 8, initial_channels // 4, stride=1),
            torch.nn.ReLU(),
            DWConv2d_BN_ReLU(initial_channels // 4, initial_channels // 2, stride=2),
            torch.nn.ReLU(),
            DWConv2d_BN_ReLU(initial_channels // 2, initial_channels, stride=1),
            torch.nn.ReLU(),
            DWConv2d_BN_ReLU(initial_channels, initial_channels, stride=1)
        )

        # 多阶段特征提取
        self.stages = nn.ModuleList()
        current_dim = initial_channels
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        # 构建MobileMamba块
        for stage_idx in range(3):
            stage = nn.Sequential()
            stage.append(
                DWConv2d_BN_ReLU(
                    in_channels = current_dim,
                    out_channels = embed_dims[stage_idx],
                    stride=2
                )
            )
            current_dim = embed_dims[stage_idx]

            for block_idx in range(depth[stage_idx]):
                stage.append(
                    MobileViMBlock(
                        type = 's',
                        ed = current_dim,
                        global_ratio = global_ratio[stage_idx],
                        local_ratio = local_ratio[stage_idx],
                        kernels = kernels[stage_idx],
                        drop_path = dprs[sum(depth[:stage_idx]) + block_idx],
                        ssm_ratio = ssm_ratio,
                        forward_type = forward_type
                    )
                )
            self.stages.append(stage)

            # --- 解码器部分 ---
            self.pixel_decoder = MSDeformAttnPixelDecoder(
                input_dims=embed_dims[::-1],
                dropout=0.0,
                num_heads=decoder_heads,
                hidden_dim=decoder_hidden_dim,
                depth=decoder_depths,
                output_dim=decoder_hidden_dim,
                enc_n_points=decoder_anchor_points,
                ffn_dim=decoder_hidden_dim * decoder_expansion,
                activation='silu',
                use_fpn=True
            )

            self.afp = AFP(
                num_resolutions=3,
                depth=afp_depths,
                pixel_dim=decoder_hidden_dim,
                latent_dim=afp_latent_dim,
                num_latents=afp_num_latents,
                num_heads=decoder_heads,
                expansion=decoder_expansion,
                activation='silu',
                norm="torchLN"
            )

            self.isd = ISD(
                num_resolutions=3,
                depth=isd_depths,
                pixel_dim=decoder_hidden_dim,
                query_dim=afp_latent_dim,
                num_heads=decoder_heads,
                output_dim=decoder_hidden_dim,
                expansion=decoder_expansion,
                activation='silu',
                norm="torchLN"
            )


    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回不需要权重衰减的参数
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        # 编码阶段
        x = self.patch_embed(x)

        enc_features = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            enc_features.append(x)

        # 反转特征顺序
        enc_features = enc_features

        # 多尺度解码
        fpn_outputs, decoder_outputs = self.pixel_decoder(enc_features)

        # 特征金字塔处理
        idrs = self.afp(decoder_outputs[:3])
        multi_scale_features = self.isd(fpn_outputs[:3], idrs)

        # 多分辨率融合
        target_size = max([f.shape[-2:] for f in multi_scale_features], key=lambda x: x[0])
        fused_features = 0
        for feat in multi_scale_features:
            resized_feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features += resized_feat
        fused_features /= len(multi_scale_features)

        return fused_features  # [B, decoder_hidden_dim, H, W]

class DepthPostProcess(nn.Module):
    """深度估计后处理模块"""
    def __init__(self, in_channels=256, upscale_factor=4):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            Conv2d_BN(in_channels, in_channels//2, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.final_conv = nn.Sequential(
            Conv2d_BN(in_channels//2, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.final_conv(x)

class ClassificationHead(nn.Module):
    """分类任务后处理模块"""
    def __init__(self, in_channels, num_classes, dropout=0.2):
        """
        Args:
            in_channels (int): 输入特征通道数（对应decoder_hidden_dim）
            num_classes (int): 分类类别数
            dropout (float): Dropout概率，默认0.2
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.flatten = nn.Flatten(1)         # 展平层
        self.dropout = nn.Dropout(dropout)    # 随机失活
        self.fc = nn.Linear(in_channels, num_classes)  # 全连接层

    def forward(self, x):
        """
        输入: [B, C, H, W]
        输出: [B, num_classes]
        """
        x = self.pool(x)    # [B, C, 1, 1]
        x = self.flatten(x) # [B, C]
        x = self.dropout(x)
        return self.fc(x)

def replace_batchnorm(net):
    # 遍历网络的所有子模块
    for child_name, child in net.named_children():
        # 如果子模块有fuse方法，则调用fuse方法进行融合
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            # 将融合后的模块替换原来的子模块
            setattr(net, child_name, fused)
            # 递归处理融合后的模块
            replace_batchnorm(fused)
        # 如果子模块是BatchNorm2d，则替换为Identity（恒等映射）
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        # 如果子模块不是BatchNorm2d且没有fuse方法，则递归处理
        else:
            replace_batchnorm(child)


def print_model_summary(model, device='cuda'):
    """支持GPU的模型分析函数"""
    # 确保模型在目标设备
    model = model.to(device)

    # 生成对应设备的输入张量
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # FLOPs计算
    flops = FlopCountAnalysis(model, input_tensor)

    print(f"Model Architecture:")
    print(model)
    print("\n" + "=" * 50)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("\nFLOPs Analysis:")
    print(flop_count_table(flops))



CFG_mobilevim_xxs = {
        'model_type':'xx_small',
        'img_size': 224,
        'embed_dims': [192, 384, 448],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0,
        'ssm_ratio': 2,
    }

CFG_mobilevim_xs = {
        'model_type':'x_small',
        'img_size': 256,
        'embed_dims': [200, 376, 448],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0,
        'ssm_ratio': 2,
    }

CFG_mobilevim_s = {
        'model_type':'small',
        'img_size': 384,
        'embed_dims': [200, 376, 448],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0,
        'ssm_ratio': 2,
    }


@MODEL.register_module
def mobilevim_xxs(num_classes=1000, pretrained=False,  fuse=False, pretrained_cfg=None, model_cfg=CFG_mobilevim_xxs):
    model = MobileViM(num_classes=num_classes, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@MODEL.register_module
def mobilevim_xs(num_classes=1000, pretrained=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_mobilevim_xs):
    model = MobileViM(num_classes=num_classes, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

@MODEL.register_module
def mobilevim_s(num_classes=1000, pretrained=False,  fuse=False, pretrained_cfg=None, model_cfg=CFG_mobilevim_s):
    model = MobileViM(num_classes=num_classes, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model


if __name__ == "__main__":
    from thop import profile, clever_format
    from torchinfo import summary
    import argparse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 配置字典更新
    CFG_mobilevim_xxs = {
     ** CFG_mobilevim_xxs,
    'decoder_hidden_dim': 256
    }

    CFG_mobilevim_xs = {
     ** CFG_mobilevim_xs,
    'decoder_hidden_dim': 384
    }

    CFG_mobilevim_s = {
     ** CFG_mobilevim_s,
    'decoder_hidden_dim': 512
    }

    # 模型配置映射
    model_configs = {
        "mobilevim_xxs": CFG_mobilevim_xxs,
        "mobilevim_xs": CFG_mobilevim_xs,
        "mobilevim_s": CFG_mobilevim_s,
    }

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=model_configs.keys(),
                        default="mobilevim_s", help="选择模型版本")
    parser.add_argument('-s', '--size', type=int, default=224, help="输入图像尺寸")
    args = parser.parse_args()


    # 构建完整模型
    class DepthEstimationModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.encoder = MobileViM(**cfg)

        def forward(self, x):
            x = self.encoder(x)
            return x

    # 实例化模型并转移到GPU
    model = DepthEstimationModel(model_configs[args.model]).to(device)

    # 创建GPU输入张量
    input_tensor = torch.randn(1, 3, args.size, args.size).to(device)

    # 计算整体参数量和FLOPs
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"\n{'=' * 30} 模型统计 {'=' * 30}")
    print(f"模型类型: {args.model}")
    print(f"输入尺寸: {args.size}x{args.size}")
    print(f"总参数量: {params}")
    print(f"总计算量: {flops}\n")
