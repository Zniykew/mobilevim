from copy import deepcopy
from typing import Callable, List, Union

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from model.ops.modules import MSDeformAttn
from idisc_utils import (Conv2d, PositionEmbeddingSine, _get_activation_fn,
                         c2_xavier_fill, get_norm)


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    """
    仅包含编码器的多尺度变形注意力Transformer模块。

    参数:
        input_dim (int): 输入特征的维度，默认为256。
        num_heads (int): 注意力头的数量，默认为8。
        depth (int): 编码器层数，默认为6。
        ffn_dim (int): 前馈网络的维度，默认为1024。
        dropout (float): Dropout概率，默认为0.0。
        activation (str): 激活函数类型，默认为"gelu"。
        num_resolutions (int): 分辨率的数量，默认为4。
        enc_n_points (int): 每个查询点的数量，默认为8。
    """
    def __init__(
        self,
        input_dim=256,
        num_heads=8,
        depth=6,
        ffn_dim=1024,
        dropout=0.0,
        activation="gelu",
        num_resolutions=4,
        enc_n_points=8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            input_dim,
            ffn_dim,
            dropout,
            activation,
            num_resolutions,
            num_heads,
            enc_n_points,
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, depth)
        self.level_embed = nn.Parameter(torch.Tensor(num_resolutions, input_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置模型参数。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        """
        计算有效区域的比例。

        参数:
            mask (torch.Tensor): 掩码张量。

        返回:
            torch.Tensor: 有效区域的比例。
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        """
        前向传播方法。

        参数:
            srcs (List[torch.Tensor]): 输入特征图列表。
            pos_embeds (List[torch.Tensor]): 位置嵌入列表。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 编码后的特征、空间形状和层级起始索引。
        """
        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
            for x in srcs
        ]
        # 准备输入给编码器
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # 编码器
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    """
    多尺度变形注意力Transformer编码器层。

    参数:
        d_model (int): 模型的维度，默认为256。
        d_ffn (int): 前馈网络的维度，默认为1024。
        dropout (float): Dropout概率，默认为0.1。
        activation (str): 激活函数类型，默认为"gelu"。
        n_levels (int): 分辨率的数量，默认为4。
        n_heads (int): 注意力头的数量，默认为8。
        n_points (int): 每个查询点的数量，默认为4。
        pre_norm (bool): 是否在自注意力之前进行归一化，默认为True。
    """
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="gelu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        pre_norm=True,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        # 自注意力
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """
        将位置嵌入添加到张量中。

        参数:
            tensor (torch.Tensor): 输入张量。
            pos (torch.Tensor): 位置嵌入张量。

        返回:
            torch.Tensor: 添加位置嵌入后的张量。
        """
        return tensor if pos is None else tensor + pos

    @autocast(enabled=False)
    def forward(
        self,
        src: torch.Tensor,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        """
        前向传播方法。

        参数:
            src (torch.Tensor): 输入特征张量。
            pos (torch.Tensor): 位置嵌入张量。
            reference_points (torch.Tensor): 参考点张量。
            spatial_shapes (torch.Tensor): 空间形状张量。
            level_start_index (torch.Tensor): 层级起始索引张量。
            padding_mask (torch.Tensor): 填充掩码张量（可选）。

        返回:
            torch.Tensor: 处理后的特征张量。
        """
        # 自注意力
        src_attn_path = src.clone()
        if self.pre_norm:
            src_attn_path = self.norm1(src_attn_path)
        src_attn_path = self.self_attn(
            self.with_pos_embed(src_attn_path, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        if not self.pre_norm:
            src_attn_path = self.norm1(src_attn_path)

        src = src + self.dropout1(src_attn_path)

        # 前馈网络
        src_ffn_path = src.clone()
        if self.pre_norm:
            src_ffn_path = self.norm2(src_ffn_path)
        src_ffn_path = self.linear2(
            self.dropout2(self.activation(self.linear1(src_ffn_path)))
        )
        if not self.pre_norm:
            src_ffn_path = self.norm2(src_ffn_path)
        src = src + self.dropout3(src_ffn_path)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    """
    多尺度变形注意力Transformer编码器。

    参数:
        encoder_layer (MSDeformAttnTransformerEncoderLayer): 编码器层实例。
        num_layers (int): 编码器层数。
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        for i in range(num_layers):
            setattr(self, f"layer_{i+1}", deepcopy(encoder_layer))
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        获取参考点。

        参数:
            spatial_shapes (List[Tuple[int, int]]): 空间形状列表。
            valid_ratios (torch.Tensor): 有效区域比例张量。
            device (torch.device): 设备。

        返回:
            torch.Tensor: 参考点张量。
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        """
        前向传播方法。

        参数:
            src (torch.Tensor): 输入特征张量。
            spatial_shapes (torch.Tensor): 空间形状张量。
            level_start_index (torch.Tensor): 层级起始索引张量。
            valid_ratios (torch.Tensor): 有效区域比例张量。
            pos (torch.Tensor): 位置嵌入张量（可选）。
            padding_mask (torch.Tensor): 填充掩码张量（可选）。

        返回:
            torch.Tensor: 编码后的特征张量。
        """
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for i in range(self.num_layers):
            output = getattr(self, f"layer_{i+1}")(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


class MSDeformAttnPixelDecoder(nn.Module):
    """
    多尺度变形注意力像素解码器。

    参数:
        input_dims (List[int]): 输入特征图的通道数列表。
        dropout (float): Dropout概率。
        num_heads (int): 注意力头的数量。
        hidden_dim (int): 隐藏层的维度。
        depth (int): 编码器层数。
        output_dim (int): 输出特征图的维度。
        norm (Union[str, Callable]): 归一化类型，默认为"BN"。
        enc_n_points (int): 每个查询点的数量，默认为4。
        activation (str): 激活函数类型，默认为"silu"。
        ffn_dim (int): 前馈网络的维度，默认为1024。
        use_fpn (bool): 是否使用FPN结构，默认为True。
    """
    def __init__(
        self,
        input_dims: List[int],
        dropout: float,
        num_heads: int,
        hidden_dim: int,
        depth: int,
        output_dim: int,
        norm: Union[str, Callable] = "BN",
        enc_n_points: int = 4,
        activation: str = "silu",
        ffn_dim: int = 1024,
        use_fpn: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.input_dims = input_dims[::-1]  # 从低分辨率到高分辨率
        self.num_resolutions = len(input_dims)
        self.use_fpn = use_fpn

        self.pos_embedding = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)

        for i, in_channels in enumerate(self.input_dims):
            setattr(
                self,
                f"input_adapter_{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(1, self.hidden_dim),
                ),
            )
            nn.init.xavier_uniform_(
                getattr(self, f"input_adapter_{i+1}")[0].weight, gain=1
            )
            nn.init.constant_(getattr(self, f"input_adapter_{i+1}")[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            input_dim=self.hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            depth=depth,
            num_resolutions=self.num_resolutions,
            enc_n_points=enc_n_points,
            activation=activation,
        )

        # 额外的FPN层
        use_bias = norm == ""
        for idx, in_channels in enumerate(self.input_dims):
            lateral_norm = get_norm(norm, self.output_dim)
            out_norm = get_norm(norm, self.output_dim)
            fpn_norm = get_norm(norm, self.output_dim)

            lateral_fpn_conv = Conv2d(
                in_channels,
                self.output_dim,
                kernel_size=1,
                bias=use_bias,
                norm=lateral_norm,
                activation=_get_activation_fn(activation),
            )
            out_conv = Conv2d(
                self.hidden_dim,
                self.output_dim,
                kernel_size=1,
                bias=use_bias,
                norm=out_norm,
                activation=_get_activation_fn(activation),
            )
            fpn_conv = Conv2d(
                self.output_dim,
                self.output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=fpn_norm,
                activation=_get_activation_fn(activation),
            )
            c2_xavier_fill(lateral_fpn_conv)
            c2_xavier_fill(out_conv)
            c2_xavier_fill(fpn_conv)
            if self.use_fpn:
                setattr(self, f"lateral_fpn_conv_{idx+1}", lateral_fpn_conv)
                setattr(self, f"fpn_conv_{idx+1}", fpn_conv)
            setattr(self, f"out_conv_{idx+1}", out_conv)

    def forward(self, features: List[torch.Tensor]):
        """
        前向传播方法。

        参数:
            features (List[torch.Tensor]): 输入特征图列表。

        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: FPN输出特征图列表和原始输出特征图列表。
        """
        srcs = []
        pos = []

        assert [feats.shape[1] for feats in features] == self.input_dims
        for idx, f in enumerate(self.input_dims):
            x = features[idx]
            srcs.append(getattr(self, f"input_adapter_{idx+1}")(x))
            pos.append(self.pos_embedding(x))

        ys_flat, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = ys_flat.shape[0]

        # 计算每个原始未展平分割的元素差异
        splits_len = [
            end_cur_split - start_cur_split
            for start_cur_split, end_cur_split in zip(
                level_start_index, [*level_start_index[1:], ys_flat.shape[1]]
            )
        ]
        assert len(splits_len) == self.num_resolutions
        ys = torch.split(ys_flat, splits_len, dim=1)

        # 不同尺度的输出
        raw_outputs = []
        for idx, y in enumerate(ys):
            y = (
                y.transpose(1, 2)
                .view(bs, -1, spatial_shapes[idx][0], spatial_shapes[idx][1])
                .contiguous()
            )
            raw_outputs.append(getattr(self, f"out_conv_{idx+1}")(y))

        # FPN层次结构：原始输出 + 编码器特征的跳接连接
        fpn_outputs = []
        if self.use_fpn:
            for idx, f in enumerate(self.input_dims):
                encoder_map = getattr(self, f"lateral_fpn_conv_{idx+1}")(features[idx])
                if idx == 0:
                    y = encoder_map + F.interpolate(
                        raw_outputs[idx], size=encoder_map.shape[-2:], mode="nearest"
                    )
                else:
                    y = (
                        encoder_map
                        + raw_outputs[idx]
                        + F.interpolate(
                            fpn_outputs[-1], size=encoder_map.shape[-2:], mode="nearest"

                        )
                    )
                y = getattr(self, f"fpn_conv_{idx+1}")(y)
                fpn_outputs.append(y)

        return fpn_outputs, raw_outputs

    @classmethod
    def build(cls, config):
        obj = cls(
            input_dims=config["model"]["pixel_encoder"]["embed_dims"],
            dropout=0.0,
            num_heads=config["model"]["pixel_decoder"]["heads"],
            hidden_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            depth=config["model"]["pixel_decoder"]["depths"],
            output_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            enc_n_points=config["model"]["pixel_decoder"]["anchor_points"],
            ffn_dim=config["model"]["pixel_decoder"]["hidden_dim"]
            * config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj
