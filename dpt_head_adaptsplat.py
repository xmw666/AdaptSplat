from typing import List, Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import create_uv_grid, position_grid_to_embed
import matplotlib.pyplot as plt
import numpy as np

try:
    from pytorch_wavelets import DWTForward
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    print("Warning: pytorch_wavelets not found. Install with: pip install pytorch_wavelets")

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FPAWeightGenerator(nn.Module):

    def __init__(self, in_channels: int):
        super(FPAWeightGenerator, self).__init__()

        mid_channels = max(in_channels // 2, 16)

        if PYTORCH_WAVELETS_AVAILABLE:
            self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        else:
            self.dwt = None

        # channel_proj: 3*in_channels -> mid_channels  (at H/8)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(mid_channels, eps=1e-6, data_format="channels_first"),
            # nn.GELU()
        )

        self.weight_conv = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False)

    def dwt2d(self, x: torch.Tensor):
        if self.dwt is None:
            raise RuntimeError("pytorch_wavelets not installed. Install with: pip install pytorch_wavelets")
        yl, yh = self.dwt(x)
        LH = yh[0][:, :, 0, :, :]  # (B, C, H/2, W/2)
        HL = yh[0][:, :, 1, :, :]
        HH = yh[0][:, :, 2, :, :]
        return LH, HL, HH

    def forward(self, feat_4x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_4x: (B, in_channels, H/4, W/4) - layer_0 原始特征
        Returns:
            weight_map: (B, 1, H/4, W/4)，值域 (0, 1)
        """
        with torch.cuda.amp.autocast(enabled=False):
            input_dtype = feat_4x.dtype
            feat_4x = feat_4x.float()

            LH, HL, HH = self.dwt2d(feat_4x) 
            hf = torch.cat([LH, HL, HH], dim=1) 
            hf = self.channel_proj(hf) 
            hf = F.interpolate(hf, scale_factor=2, mode='bilinear', align_corners=False)  # (B, mid, H/4, W/4)
            weight_map = torch.sigmoid(self.weight_conv(hf))  # (B, 1, H/4, W/4)
            weight_map = weight_map.to(input_dtype)

        return weight_map


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: List[int] = [256, 512, 1024, 2048],
        features: int = 1024,
        out_channels: List[int] = [256, 512, 1024, 2048],
        use_sda: bool = True,
    ) -> None:
        super(DPTHead, self).__init__()

        # Must be 4 layers
        assert len(dim_in) == 4, f"This DPT head requires 4 layers, got {len(dim_in)}"

        # Layer normalization for each stage
        self.norm0 = nn.LayerNorm(dim_in[0])
        self.norm1 = nn.LayerNorm(dim_in[1])
        self.norm2 = nn.LayerNorm(dim_in[2])
        self.norm3 = nn.LayerNorm(dim_in[3])
        # Scratch modules for feature projection
        self.scratch = _make_scratch(out_channels, features, num_layers=4)

        self.scratch.refinenet0 = _make_fusion_block(features)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

        self.use_sda = use_sda
        if self.use_sda:
            self.hfpe_weight_gen0 = FPAWeightGenerator(in_channels=dim_in[0])
            self.hfpe_weight_gen1 = FPAWeightGenerator(in_channels=dim_in[1])
            # 每层独立的可学习缩放因子
            self.gamma_param0 = nn.Parameter(torch.tensor([0.01]))
            self.gamma_param1 = nn.Parameter(torch.tensor([0.01]))
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        image_sizes: List[int],
        patch_size: int,
    ) -> torch.Tensor:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
                [stage0_tokens, stage1_tokens, stage2_tokens, stage3_tokens]
            image_sizes (List[int]): Image sizes [H, W].
            patch_size (int): Patch size used in the transformer.

        Returns:
            Tensor: Fused feature map with shape [B, (H*W), C]
        """
        H, W = image_sizes

        # For 4 layers: stage0 (H/4), stage1 (H/8), stage2 (H/16), stage3 (H/32)
        hh0 = H // 4
        ww0 = W // 4
        hh1 = H // 8
        ww1 = W // 8
        hh2 = H // 16
        ww2 = W // 16
        hh3 = H // 32
        ww3 = W // 32

        out = []

        # Stage 0: /4 resolution (shallowest)
        x0 = self.norm0(aggregated_tokens_list[0])
        x0 = x0.permute(0, 2, 1).reshape((x0.shape[0], x0.shape[-1], hh0, ww0))
        x0 = self._apply_pos_embed(x0, W, H)
        out.append(x0)

        # Stage 1: /8 resolution
        x1 = self.norm1(aggregated_tokens_list[1])
        x1 = x1.permute(0, 2, 1).reshape((x1.shape[0], x1.shape[-1], hh1, ww1))
        x1 = self._apply_pos_embed(x1, W, H)
        out.append(x1)

        # Stage 2: /16 resolution
        x2 = self.norm2(aggregated_tokens_list[2])
        x2 = x2.permute(0, 2, 1).reshape((x2.shape[0], x2.shape[-1], hh2, ww2))
        x2 = self._apply_pos_embed(x2, W, H)
        out.append(x2)

        # Stage 3: /32 resolution (deepest)
        x3 = self.norm3(aggregated_tokens_list[3])
        x3 = x3.permute(0, 2, 1).reshape((x3.shape[0], x3.shape[-1], hh3, ww3))
        x3 = self._apply_pos_embed(x3, W, H)
        out.append(x3)

        out = self.scratch_forward(out)
        out = self._apply_pos_embed(out, W, H)

        return rearrange(out, "b c h w -> b (h w) c")

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        
        layer_0, layer_1, layer_2, layer_3 = features

        layer_0_rn = self.scratch.layer0_rn(layer_0)  
        layer_1_rn = self.scratch.layer1_rn(layer_1) 
        layer_2_rn = self.scratch.layer2_rn(layer_2)  
        layer_3_rn = self.scratch.layer3_rn(layer_3)  

        if self.use_sda:
            hfpe_weight0 = self.hfpe_weight_gen0(layer_0)  # (B, 1, H/4, W/4)
            hfpe_weight1 = self.hfpe_weight_gen1(layer_1)  # (B, 1, H/8, W/8)
        out = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        if self.use_sda:
            shallow_1_processed = self.scratch.refinenet1.resConfUnit1(layer_1_rn)
            fused_1 = out + shallow_1_processed * (1.0 + self.gamma_param1 * hfpe_weight1)
            out = self.scratch.refinenet1.resConfUnit2(fused_1)
            out = custom_interpolate(
                out,
                size=layer_0_rn.shape[2:],
                mode="bilinear",
                align_corners=self.scratch.refinenet1.align_corners
            )
            out = self.scratch.refinenet1.out_conv(out)
        else:
            out = self.scratch.refinenet1(out, layer_1_rn, size=layer_0_rn.shape[2:])
        del layer_1_rn, layer_1

        if self.use_sda:
            shallow_0_processed = self.scratch.refinenet0.resConfUnit1(layer_0_rn)
            fused_0 = out + shallow_0_processed * (1.0 + self.gamma_param0 * hfpe_weight0)
            out = self.scratch.refinenet0.resConfUnit2(fused_0)
            out = self.scratch.refinenet0.out_conv(out)
        else:
            out = self.scratch.refinenet0(out, layer_0_rn)
        del layer_0_rn, layer_0

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, num_layers: int = 4, groups: int = 1) -> nn.Module:
    scratch = nn.Module()
    out_shape0 = out_shape
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape

    assert num_layers == 4, "This scratch module only supports 4 layers"

    scratch.layer0_rn = nn.Conv2d(
        in_shape[0], out_shape0, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer1_rn = nn.Conv2d(
        in_shape[1], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[2], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[3], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if size is not None:
            modifier = {"size": size}
            output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)