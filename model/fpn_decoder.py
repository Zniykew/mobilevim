"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch import nn
from torch.nn import functional as F

from idisc_utils import Conv2d, c2_xavier_fill, get_norm


class BasePixelDecoder(nn.Module):
    """
    基础像素解码器模块。

    参数:
        input_dims (List[int]): 输入特征图的通道数列表。
        hidden_dim (int): 隐藏层的维度。
        output_dim (int): 输出特征图的维度。
        norm (Union[str, Callable]): 归一化类型，默认为"BN"（Batch Normalization）。
        **kwargs: 其他可选参数。
    """
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int,
        output_dim: int,
        norm: Union[str, Callable] = "BN",
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.in_features = input_dims[::-1]  # 反转输入特征图的顺序，从低分辨率到高分辨率
        use_bias = norm == ""  # 如果归一化为空字符串，则使用偏置

        # 构建每一层的卷积模块
        for idx, in_channels in enumerate(self.in_features):
            if idx == 0:
                # 第一层直接将输入特征图转换为目标维度
                output_norm = get_norm(norm, output_dim)
                output_conv = Conv2d(
                    in_channels,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                c2_xavier_fill(output_conv)  # 初始化权重
                self.add_module("layer_{}".format(idx + 1), output_conv)
            else:
                # 后续层通过横向连接和上采样实现FPN结构
                lateral_norm = get_norm(norm, hidden_dim)
                output_norm = get_norm(norm, output_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    hidden_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                c2_xavier_fill(lateral_conv)
                c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

    def forward(self, features):
        """
        前向传播方法。

        参数:
            features (List[torch.Tensor]): 输入特征图列表。

        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: FPN输出特征图列表和相同特征图列表。
        """
        fpn_output = []
        # 将输入特征图反转为从低分辨率到高分辨率的顺序
        for idx, f in enumerate(self.in_features):
            x = features[idx]
            if idx == 0:
                # 第一层直接处理输入特征图
                y = getattr(self, f"layer_{idx+1}")(x)
            else:
                # 后续层通过横向连接和上采样实现FPN结构
                cur_fpn = getattr(self, f"adapter_{idx+1}")(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = getattr(self, f"layer_{idx+1}")(y)
            fpn_output.append(y)
        return fpn_output, fpn_output

    @classmethod
    def build(cls, config):
        """
        根据配置构建BasePixelDecoder模块。

        参数:
            config (Dict): 配置字典。

        返回:
            BasePixelDecoder: 构建好的BasePixelDecoder模块实例。
        """
        obj = cls(
            input_dims=config["model"]["pixel_encoder"]["embed_dims"],
            hidden_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            output_dim=config["model"]["pixel_decoder"]["hidden_dim"],
        )
        return obj
