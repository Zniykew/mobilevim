"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.defattn_decoder import MSDeformAttnPixelDecoder
from model.fpn_decoder import BasePixelDecoder
from model.id_module import AFP, ISD


class IDisc(nn.Module):
    """
    IDisc 模型的主类。

    参数:
        pixel_encoder (nn.Module): 像素编码器模块。
        afp (nn.Module): AFP（Adaptive Feature Pyramid）模块。
        pixel_decoder (nn.Module): 像素解码器模块。
        isd (nn.Module): ISD（Implicit Surface Decoder）模块。
        loss (nn.Module): 损失函数模块。
        afp_min_resolution (int): AFP 模块使用的最小分辨率索引，默认为1。
        eps (float): 小数值，用于数值稳定性，默认为1e-6。
        **kwargs: 其他可选参数。
    """

    def __init__(
        self,
        pixel_encoder: nn.Module,
        afp: nn.Module,
        pixel_decoder: nn.Module,
        isd: nn.Module,
        loss: nn.Module,
        afp_min_resolution=1,
        eps: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.pixel_encoder = pixel_encoder
        self.afp = afp
        self.pixel_decoder = pixel_decoder
        self.isd = isd
        self.afp_min_resolution = afp_min_resolution
        self.loss = loss

    def invert_encoder_output_order(
        self, xs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        反转编码器输出的顺序。

        参数:
            xs (Tuple[torch.Tensor, ...]): 编码器的输出张量列表。

        返回:
            Tuple[torch.Tensor, ...]: 反转后的张量列表。
        """
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(
        self, decoder_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        过滤解码器相关的分辨率。

        参数:
            decoder_outputs (Tuple[torch.Tensor, ...]): 解码器的输出张量列表。

        返回:
            Tuple[torch.Tensor, ...]: 过滤后的张量列表。
        """
        return tuple(decoder_outputs[self.afp_min_resolution :])

    def forward(
        self,
        image: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        前向传播方法。

        参数:
            image (torch.Tensor): 输入图像张量。
            gt (Optional[torch.Tensor]): 地面真值张量（可选）。
            mask (Optional[torch.Tensor]): 掩码张量（可选）。

        返回:
            Tuple[torch.Tensor, Dict[str, Dict[str, Any]], Dict[str, Any]]: 输出张量、损失字典和其他信息。
        """
        losses = {"opt": {}, "stat": {}}
        original_shape = gt.shape[-2:] if gt is not None else image.shape[-2:]

        # 使用像素编码器生成特征图
        encoder_outputs = self.pixel_encoder(image)
        encoder_outputs = self.invert_encoder_output_order(encoder_outputs)

        # 使用像素解码器处理特征图，并过滤有用的分辨率
        fpn_outputs, decoder_outputs = self.pixel_decoder(encoder_outputs)

        decoder_outputs = self.filter_decoder_relevant_resolutions(decoder_outputs)
        fpn_outputs = self.filter_decoder_relevant_resolutions(fpn_outputs)

        # 使用 AFP 和 ISD 模块生成最终输出
        idrs = self.afp(decoder_outputs)
        outs = self.isd(fpn_outputs, idrs)

        out_lst = []
        for out in outs:
            if out.shape[1] == 1:
                out = F.interpolate(
                    torch.exp(out),
                    size=outs[-1].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                out = self.normalize_normals(
                    F.interpolate(
                        out,
                        size=outs[-1].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            out_lst.append(out)

        out = F.interpolate(
            torch.mean(torch.stack(out_lst, dim=0), dim=0),
            original_shape,
            mode="bilinear" if out.shape[1] == 1 else "bicubic",
            align_corners=True,
        )

        if gt is not None:
            losses["opt"] = {
                self.loss.name: self.loss.weight
                * self.loss(out, target=gt, mask=mask.bool(), interpolate=True)
            }
        return (
            out if out.shape[1] == 1 else out[:, :3],
            losses,
            {"outs": outs, "queries": idrs},
        )

    def normalize_normals(self, norms):
        """
        归一化法线张量。

        参数:
            norms (torch.Tensor): 输入法线张量。

        返回:
            torch.Tensor: 归一化后的法线张量。
        """
        min_kappa = 0.01
        norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
        norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
        kappa = F.elu(kappa) + 1.0 + min_kappa
        norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
        return norms

    def load_pretrained(self, model_file):
        """
        加载预训练模型权重。

        参数:
            model_file (str): 预训练模型文件路径。
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )
        self.load_state_dict(new_state_dict)

    def get_params(self, config):
        """
        获取模型参数和学习率。

        参数:
            config (Dict[str, Any]): 配置字典。

        返回:
            Tuple[List[Dict[str, Any]], List[float]]: 参数组和对应的最大学习率。
        """
        backbone_lr = config["model"]["pixel_encoder"].get(
            "lr_dedicated", config["training"]["lr"] / 10
        )
        params = [
            {"params": self.pixel_decoder.parameters()},
            {"params": self.afp.parameters()},
            {"params": self.isd.parameters()},
            {"params": self.pixel_encoder.parameters()},
        ]
        max_lrs = [config["training"]["lr"]] * 3 + [backbone_lr]
        return params, max_lrs

    @property
    def device(self):
        """
        获取模型所在的设备。

        返回:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):
        """
        根据配置构建 IDisc 模型。

        参数:
            config (Dict[str, Dict[str, Any]]): 配置字典。

        返回:
            IDisc: 构建好的 IDisc 模型实例。
        """
        pixel_encoder_img_size = config["model"]["pixel_encoder"]["img_size"]
        pixel_encoder_pretrained = config["model"]["pixel_encoder"].get(
            "pretrained", None
        )
        config_backone = {"img_size": np.array(pixel_encoder_img_size)}
        if pixel_encoder_pretrained is not None:
            config_backone["pretrained"] = pixel_encoder_pretrained

        mod = importlib.import_module("idisc.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder = pixel_encoder_factory(**config_backone)

        pixel_encoder_embed_dims = getattr(pixel_encoder, "embed_dims")
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims

        pixel_decoder = (
            MSDeformAttnPixelDecoder.build(config)
            if config["model"]["attn_dec"]
            else BasePixelDecoder.build(config)
        )
        afp = AFP.build(config)
        isd = ISD.build(config)

        mod = importlib.import_module("idisc.optimization.losses")
        loss = getattr(mod, config["training"]["loss"]["name"]).build(config)

        return deepcopy(
            cls(
                pixel_encoder=pixel_encoder,
                pixel_decoder=pixel_decoder,
                afp=afp,
                isd=isd,
                loss=loss,
                afp_min_resolution=len(pixel_encoder_embed_dims)
                - config["model"]["isd"]["num_resolutions"],
            )
        )
