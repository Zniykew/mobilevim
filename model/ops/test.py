# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import time

import torch
import torch.nn as nn
from functions.ms_deform_attn_func import (MSDeformAttnFunction,
                                           ms_deform_attn_core_pytorch)
from torch.autograd import gradcheck

# 定义一些常量
N, M, D = 1, 2, 2  # 批次大小、多尺度采样点数量、特征维度
Lq, L, P = 2, 2, 2  # 查询点数量、每个查询点的采样点数量、每个采样点的位置数量
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()  # 特征图的尺寸
level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))  # 每个特征图的起始索引
S = sum([(H * W).item() for H, W in shapes])  # 所有特征图的总像素数量

torch.manual_seed(3)  # 设置随机种子以保证结果可重复


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    # 生成随机输入数据
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)  # 归一化注意力权重
    im2col_step = 2  # 每次处理的图像块大小

    # 使用PyTorch实现计算输出
    output_pytorch = (
        ms_deform_attn_core_pytorch(
            value.double(),
            shapes,
            sampling_locations.double(),
            attention_weights.double(),
        )
        .detach()
        .cpu()
    )
    # 使用CUDA实现计算输出
    output_cuda = (
        MSDeformAttnFunction.apply(
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        )
        .detach()
        .cpu()
    )
    # 检查两个输出是否接近
    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()  # 最大绝对误差
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()  # 最大相对误差

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    # 生成随机输入数据
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)  # 归一化注意力权重
    im2col_step = 2  # 每次处理的图像块大小

    # 使用PyTorch实现计算输出
    output_pytorch = (
        ms_deform_attn_core_pytorch(
            value, shapes, sampling_locations, attention_weights
        )
        .detach()
        .cpu()
    )
    # 使用CUDA实现计算输出
    output_cuda = (
        MSDeformAttnFunction.apply(
            value,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        .detach()
        .cpu()
    )
    # 检查两个输出是否接近
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()  # 最大绝对误差
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()  # 最大相对误差

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )


def check_gradient_numerical(
    channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    # 生成随机输入数据
    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)  # 归一化注意力权重
    im2col_step = 2  # 每次处理的图像块大小
    func = MSDeformAttnFunction.apply

    # 设置需要计算梯度的张量
    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    # 使用数值方法检查梯度是否正确
    gradok = gradcheck(
        func,
        (
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        ),
    )

    print(f"* {gradok} check_gradient_numerical(D={channels})")


if __name__ == "__main__":
    # 检查前向传播是否正确（双精度）
    check_forward_equal_with_pytorch_double()
    # 检查前向传播是否正确（单精度）
    check_forward_equal_with_pytorch_float()

    # 检查不同通道数量下的梯度是否正确
    for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
        check_gradient_numerical(channels, True, True, True)
