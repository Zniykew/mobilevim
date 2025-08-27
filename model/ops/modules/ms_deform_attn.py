# ------------------------------------------------------------------------------------------------
# 可变形DETR (Deformable DETR)
# 版权所有 (c) 2020 SenseTime. 保留所有权利。
# 根据 Apache License, Version 2.0 发布 [详见 LICENSE 文件]
# ------------------------------------------------------------------------------------------------
# 修改自 https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    """
    检查一个数是否是2的幂次方。
    :param n: 输入整数
    :return: 如果是2的幂次方返回True，否则返回False
    """
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("无效输入：{} (类型: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力模块 (Multi-Scale Deformable Attention Module)
    :param d_model: 隐藏维度大小
    :param n_levels: 特征层级数量
    :param n_heads: 注意力头的数量
    :param n_points: 每个注意力头在每个特征层级上的采样点数量
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # 检查 d_model 是否能被 n_heads 整除
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除，但得到 {} 和 {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # 建议将每个注意力头的维度设置为2的幂次方，以提高CUDA实现的效率
        if not _is_power_of_2(_d_per_head):
            warnings.warn("建议将 d_model 设置为使每个注意力头的维度为2的幂次方，以提高CUDA实现的效率。")

        self.im2col_step = 64  # im2col 步长

        self.d_model = d_model  # 隐藏维度大小
        self.n_levels = n_levels  # 特征层级数量
        self.n_heads = n_heads  # 注意力头的数量
        self.n_points = n_points  # 每个注意力头在每个特征层级上的采样点数量

        # 定义线性层
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()  # 初始化参数

    def _reset_parameters(self):
        """
        初始化模型参数。
        """
        # 初始化 sampling_offsets 的权重为0
        constant_(self.sampling_offsets.weight.data, 0.0)
        # 初始化 sampling_offsets 的偏置
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # 初始化 attention_weights 的权重和偏置为0
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        # 使用 Xavier 初始化 value_proj 和 output_proj 的权重
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        前向传播函数。
        :param query: 查询张量，形状为 (N, Length_{query}, C)
        :param reference_points: 参考点张量，形状为 (N, Length_{query}, n_levels, 2) 或 (N, Length_{query}, n_levels, 4)
        :param input_flatten: 展平后的输入张量，形状为 (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes: 输入的空间形状，形状为 (n_levels, 2)，表示 [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index: 每个层级的起始索引，形状为 (n_levels, )
        :param input_padding_mask: 输入的填充掩码，形状为 (N, \sum_{l=0}^{L-1} H_l \cdot W_l)，True 表示填充元素，False 表示非填充元素
        :return: 输出张量，形状为 (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 确保输入的空间形状总和等于展平后的输入长度
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # 计算 value 投影
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # 计算采样偏移量和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # 计算采样位置
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError("reference_points 的最后一维必须是2或4，但得到 {}。".format(reference_points.shape[-1]))

        # 调用 CUDA 实现的多尺度可变形注意力函数
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)  # 输出投影
        return output
