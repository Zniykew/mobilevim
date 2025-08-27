"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from idisc_utils import (AttentionLayer, PositionEmbeddingSine,
                         _get_activation_cls, get_norm)


class ISDHead(nn.Module):
    """
    ISD（Implicit Surface Decoder）头部模块。

    参数:
        depth (int): 注意力层的深度。
        pixel_dim (int): 像素特征的维度，默认为256。
        query_dim (int): 查询特征的维度，默认为256。
        num_heads (int): 注意力头的数量，默认为4。
        output_dim (int): 输出特征的维度，默认为1。
        expansion (int): MLP 层的扩展因子，默认为2。
        activation (str): 激活函数类型，默认为"silu"。
        norm (str): 归一化层类型，默认为"LN"（Layer Normalization）。
        eps (float): 小数值，用于数值稳定性，默认为1e-6。
    """
    def __init__(
        self,
        depth: int,
        pixel_dim: int = 256,
        query_dim: int = 256,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "LN",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.depth = depth
        self.eps = eps
        self.pixel_pe = PositionEmbeddingSine(pixel_dim // 2, normalize=True)

        # 构建多层交叉注意力和MLP模块
        for i in range(self.depth):
            setattr(
                self,
                f"cross_attn_{i+1}",
                AttentionLayer(
                    sink_dim=pixel_dim,
                    hidden_dim=pixel_dim,
                    source_dim=query_dim,
                    output_dim=pixel_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    pre_norm=True,
                    sink_competition=False,
                ),
            )
            setattr(
                self,
                f"mlp_{i+1}",
                nn.Sequential(
                    get_norm(norm, pixel_dim),
                    nn.Linear(pixel_dim, expansion * pixel_dim),
                    _get_activation_cls(activation),
                    nn.Linear(expansion * pixel_dim, pixel_dim),
                ),
            )

        # 输出投影层
        setattr(
            self,
            "proj_output",
            nn.Sequential(
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, pixel_dim),
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, output_dim),
            ),
        )

    def forward(self, feature_map: torch.Tensor, idrs: torch.Tensor):
        """
        前向传播方法。

        参数:
            feature_map (torch.Tensor): 输入特征图。
            idrs (torch.Tensor): 输入的IDRs（Implicit Dense Representations）。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        b, c, h, w = feature_map.shape
        feature_map = rearrange(
            feature_map + self.pixel_pe(feature_map), "b c h w -> b (h w) c"
        )

        # 逐层处理特征图
        for i in range(self.depth):
            update = getattr(self, f"cross_attn_{i+1}")(feature_map.clone(), idrs)
            feature_map = feature_map + update
            feature_map = feature_map + getattr(self, f"mlp_{i+1}")(feature_map.clone())

        out = getattr(self, "proj_output")(feature_map)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        return out


class ISD(nn.Module):
    """
    ISD（Implicit Surface Decoder）模块。

    参数:
        num_resolutions (int): 分辨率的数量。
        depth (int): 每个分辨率对应的ISDHead的深度。
        pixel_dim (int): 像素特征的维度，默认为128。
        query_dim (int): 查询特征的维度，默认为128。
        num_heads (int): 注意力头的数量，默认为4。
        output_dim (int): 输出特征的维度，默认为1。
        expansion (int): MLP 层的扩展因子，默认为2。
        activation (str): 激活函数类型，默认为"silu"。
        norm (str): 归一化层类型，默认为"torchLN"。
    """
    def __init__(
        self,
        num_resolutions,
        depth,
        pixel_dim=128,
        query_dim=128,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "torchLN",
    ):
        super().__init__()
        self.num_resolutions = num_resolutions

        # 构建多个ISDHead模块
        for i in range(num_resolutions):
            setattr(
                self,
                f"head_{i+1}",
                ISDHead(
                    depth=depth,
                    pixel_dim=pixel_dim,
                    query_dim=query_dim,
                    num_heads=num_heads,
                    output_dim=output_dim,
                    expansion=expansion,
                    activation=activation,
                    norm=norm,
                ),
            )

    def forward(
        self, xs: Tuple[torch.Tensor, ...], idrs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播方法。

        参数:
            xs (Tuple[torch.Tensor, ...]): 输入特征图列表。
            idrs (Tuple[torch.Tensor, ...]): 输入的IDRs列表。

        返回:
            Tuple[torch.Tensor, ...]: 处理后的输出张量列表。
        """
        outs = []
        for i in range(self.num_resolutions):
            out = getattr(self, f"head_{i+1}")(xs[i], idrs[i])
            outs.append(out)
        return tuple(outs)

    @classmethod
    def build(cls, config):
        """
        根据配置构建ISD模块。

        参数:
            config (Dict): 配置字典。

        返回:
            ISD: 构建好的ISD模块实例。
        """
        obj = cls(
            num_resolutions=config["model"]["isd"]["num_resolutions"],
            depth=config["model"]["isd"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            query_dim=config["model"]["afp"]["latent_dim"],
            output_dim=config["model"]["output_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj


class AFP(nn.Module):
    """
    AFP（Adaptive Feature Pyramid）模块。

    参数:
        num_resolutions (int): 分辨率的数量。
        depth (int): 注意力迭代的深度，默认为3。
        pixel_dim (int): 像素特征的维度，默认为256。
        latent_dim (int): 隐变量的维度，默认为256。
        num_latents (int): 隐变量的数量，默认为128。
        num_heads (int): 注意力头的数量，默认为4。
        activation (str): 激活函数类型，默认为"silu"。
        norm (str): 归一化层类型，默认为"torchLN"。
        expansion (int): MLP 层的扩展因子，默认为2。
        eps (float): 小数值，用于数值稳定性，默认为1e-6。
    """
    def __init__(
        self,
        num_resolutions: int,
        depth: int = 3,
        pixel_dim: int = 256,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.iters = depth
        self.num_slots = num_latents
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        self.eps = eps

        bottlenck_dim = expansion * latent_dim

        # 构建位置编码和隐变量初始化参数
        for i in range(self.num_resolutions):
            setattr(
                self,
                f"pixel_pe_{i+1}",
                PositionEmbeddingSine(pixel_dim // 2, normalize=True),
            )
            setattr(
                self,
                f"mu_{i+1}",
                nn.Parameter(torch.randn(1, self.num_slots, latent_dim)),
            )

        # 构建多层交叉注意力和MLP模块
        for j in range(self.iters):
            for i in range(self.num_resolutions):
                setattr(
                    self,
                    f"cross_attn_{i+1}_d{1}",
                    AttentionLayer(
                        sink_dim=latent_dim,
                        hidden_dim=latent_dim,
                        source_dim=pixel_dim,
                        output_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=0.0,
                        pre_norm=True,
                        sink_competition=True,
                    ),
                )
                setattr(
                    self,
                    f"mlp_cross_{i+1}_d{1}",
                    nn.Sequential(
                        get_norm(norm, latent_dim),
                        nn.Linear(latent_dim, bottlenck_dim),
                        _get_activation_cls(activation),
                        nn.Linear(bottlenck_dim, latent_dim),
                    ),
                )

    def forward(
        self, feature_maps: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播方法。

        参数:
            feature_maps (Tuple[torch.Tensor, ...]): 输入特征图列表。

        返回:
            Tuple[torch.Tensor, ...]: 处理后的隐变量列表。
        """
        b, *_ = feature_maps[0].shape
        idrs = []
        feature_maps_flat = []

        # 预处理特征图并生成隐变量
        for i in range(self.num_resolutions):
            feature_map, (h, w) = feature_maps[i], feature_maps[i].shape[-2:]
            feature_maps_flat.append(
                rearrange(
                    feature_map + getattr(self, f"pixel_pe_{i+1}")(feature_map),
                    "b d h w-> b (h w) d",
                )
            )
            idrs.append(getattr(self, f"mu_{i+1}").expand(b, -1, -1))

        # 逐层处理隐变量
        for i in range(self.num_resolutions):
            for _ in range(self.iters):
                idrs[i] = idrs[i] + getattr(self, f"cross_attn_{i+1}_d{1}")(
                    idrs[i].clone(), feature_maps_flat[i]
                )
                idrs[i] = idrs[i] + getattr(self, f"mlp_cross_{i+1}_d{1}")(
                    idrs[i].clone()
                )

        return tuple(idrs)

    @classmethod
    def build(cls, config):
        """
        根据配置构建AFP模块。

        参数:
            config (Dict): 配置字典。

        返回:
            AFP: 构建好的AFP模块实例。
        """
        output_num_resolutions = (
            len(config["model"]["pixel_encoder"]["embed_dims"])
            - config["model"]["afp"]["context_low_resolutions_skip"]
        )
        obj = cls(
            num_resolutions=output_num_resolutions,
            depth=config["model"]["afp"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            num_latents=config["model"]["afp"]["num_latents"],
            latent_dim=config["model"]["afp"]["latent_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj
