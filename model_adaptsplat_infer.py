
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict
import os
import sys
from gsplat import rasterization
from transformer import TransformerBlock, SelfAttention, MLP
from dpt_head_adaptsplat import DPTHead
from prope_custom import PropeDotProductAttention
from utils import compute_rays, compute_plucmap
from loss import LossComputer
from torch_impl import _spherical_harmonics

class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        # rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        packed=False,
                                        absgrad=False,
                                        sparse_grad=False,                                        
                                        render_mode="RGB",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 3) 
        return rendering # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            B, V, _ = test_intr.shape
            renderings = torch.zeros(B, V, H, W, 3).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv:iv+1] = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib,iv], 
                                                                      test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib,iv], 
                                                        test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
                    rendering.backward(grad_output[ib, iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None

try:
    from pytorch_wavelets import DWTForward 
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    print("Warning: pytorch_wavelets not found. Install with: pip install pytorch_wavelets")
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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


class RGBStem(nn.Module):
    """
    Standard 3-channel RGB Stem using pretrained ConvNeXt weights
    - image: 3 channels (RGB, ImageNet normalized)
    """
    def __init__(self, pretrained_stem=None):
        super().__init__()
        # ImageNet normalization for RGB channels
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.stem = pretrained_stem
        
    def forward(self, image):
        """
        Args:
            image: (B, V, 3, H, W) - in range [0, 1] or [-1, 1]
        Returns:
            (B*V, output_dim, H/4, W/4)
        """
        B, V = image.shape[:2]

        # Normalize image (ImageNet normalization)
        # If input is in range [-1, 1], map to [0, 1]
        if image.min() < 0:
            image = (image + 1) * 0.5

        image = (image - self.mean) / self.std

        # Reshape to (B*V, 3, H, W)
        x = rearrange(image, 'b v c h w -> (b v) c h w')

        # Apply conv and norm
        x = self.stem(x)

        return x



class ConvToToken(nn.Module):
    """Convert ConvNeXt feature map to MVP token format"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x, B, V):
        """
        Args:
            x: (B*V, C, H, W)
        Returns:
            (B, V, H*W, C)
        """
        BV, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B*V, H*W, C)
        x = rearrange(x, '(b v) l c -> b v l c', b=B, v=V)
        # x = self.norm(x)
        return x


class TokenToConv(nn.Module):
    """Convert MVP tokens back to ConvNeXt feature map with LayerNorm"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # LayerNorm for transformer output (channels_first format)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x, H, W):
        """
        Args:
            x: (B, V, L, C) where L=H*W
        Returns:
            (B*V, C, H, W) - normalized features
        """
        x = rearrange(x, 'b v (h w) c -> (b v) c h w', h=H, w=W)
        x = self.norm(x)
        return x


def downsample_ray_features(ray_o, ray_d, o_cross_d, target_h, target_w):
    """
    Downsample ray features to target resolution using average pooling

    Args:
        ray_o: (B, V, 3, H, W)
        ray_d: (B, V, 3, H, W)
        o_cross_d: (B, V, 3, H, W)
        target_h: target height
        target_w: target width

    Returns:
        ray_features: (B*V, 9, target_h, target_w)
    """
    B, V, _, H, W = ray_o.shape

    # Concatenate ray features: [ray_o, ray_d, o_cross_d] -> (B, V, 9, H, W)
    ray_features = torch.cat([ray_o, ray_d, o_cross_d], dim=2)

    # Reshape to (B*V, 9, H, W)
    ray_features = rearrange(ray_features, 'b v c h w -> (b v) c h w')

    # Downsample using average pooling
    if H != target_h or W != target_w:
        ray_features = F.adaptive_avg_pool2d(ray_features, (target_h, target_w))

    return ray_features


class FPA(nn.Module):

    def __init__(self, in_channels, out_channels, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

        # 使用 pytorch_wavelets 库进行 2D 离散小波变换
        if PYTORCH_WAVELETS_AVAILABLE:
            self.dwt = DWTForward(J=1, wave=wavelet, mode='zero')
        else:
            self.dwt = None

        hf_channels = in_channels * 3

        self.pixel_unshuffle = nn.MaxPool2d(kernel_size=4, stride=4)

        self.channel_proj = nn.Sequential(
            nn.Conv2d(hf_channels, out_channels, kernel_size=1, padding=0, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
        )

    def dwt2d(self, x):

        if self.dwt is None:
            raise RuntimeError("pytorch_wavelets not installed. Install with: pip install pytorch_wavelets")

       
        yl, yh = self.dwt(x)


        LH = yh[0][:, :, 0, :, :]  
        HL = yh[0][:, :, 1, :, :]  
        HH = yh[0][:, :, 2, :, :]  

        return LH, HL, HH

    def forward(self, feat_4x):

        # pytorch_wavelets 不支持混合精度，禁用 autocast
        with torch.cuda.amp.autocast(enabled=False):
            input_dtype = feat_4x.dtype

            feat_4x = feat_4x.float()

            LH, HL, HH = self.dwt2d(feat_4x) 

            hf_features = torch.cat([LH, HL, HH], dim=1)  

            hf_downsampled = self.pixel_unshuffle(hf_features) 

            hf_pe = self.channel_proj(hf_downsampled) 


            hf_pe = hf_pe.to(input_dtype)

        return hf_pe


class SelfAttentionWithHFPE(SelfAttention):

    def __init__(self, dim, head_dim, use_qk_norm=True, causal=False, bias=False):
        super().__init__(dim, head_dim, use_qk_norm, causal, bias)

    def forward(self, x, prope, stage, info, hf_pe=None):

        # QKV 投影
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)

        if hf_pe is not None:
            hf_pe_reshaped = rearrange(hf_pe, "b l (nh dh) -> b nh l dh", dh=self.head_dim)
            q = q + hf_pe_reshaped
            k = k + hf_pe_reshaped

        # QK Normalization
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if stage == 1:
            x = F.scaled_dot_product_attention(q, k, v)
        elif stage == 2:
            if prope:
                w2c = info["w2c"]
                Ks = info["Ks"]
                attn_fn = info["attn2"]
                x = attn_fn(q, k, v, viewmats=w2c, Ks=Ks)
            else:
                attn_fn = info["attn2"]
                x = attn_fn(q, k, v, viewmats=None, Ks=None)
        elif stage == 3:
            if prope:
                w2c = info["w2c"]
                Ks = info["Ks"]
                attn_fn = info["attn3"]
                x = attn_fn(q, k, v, viewmats=w2c, Ks=Ks)
            else:
                attn_fn = info["attn3"]
                x = attn_fn(q, k, v, viewmats=None, Ks=None)

        x = rearrange(x, "b nh l dh -> b l (nh dh)")
        x = self.c_proj(x)
        return x


class TransformerBlockWithHFPE(TransformerBlock):

    def __init__(self, dim, bias, head_dim, inter_multi, use_qk_norm):
        # 不调用父类 __init__，重新初始化
        nn.Module.__init__(self)  # 调用 nn.Module 的 __init__

        self.ln1 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.attn = SelfAttentionWithHFPE(
            dim=dim, bias=bias, head_dim=head_dim, use_qk_norm=use_qk_norm)

        self.ln2 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.mlp = MLP(dim=dim, bias=bias, inter_multi=inter_multi)

    def forward(self, x, prope, stage, info, hf_pe=None):

        x = x + self.attn(self.ln1(x), prope, stage, info, hf_pe)
        x = x + self.mlp(self.ln2(x))
        return x


class ChannelAlignmentModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data[:, -9:, :, :] *= 0.01 
    def forward(self, x):
        """
        Args:
            x: (B*V, C+9, H, W)
        Returns:
            (B*V, C, H, W)
        """
        x = self.proj(x)
        return x


class HybridStage(nn.Module):

    def __init__(self, convnext_stage, mvp_stage, dim):
        super().__init__()
        self.convnext_stage = convnext_stage
        self.mvp_stage = mvp_stage
        self.dim = dim

        self.conv_to_token = ConvToToken(dim)
        self.token_to_conv = TokenToConv(dim)  


    def forward(self, x, B, V, H, W, info, register_tokens=None, hf_pe=None):
     
        conv_out = torch.utils.checkpoint.checkpoint(
            self.convnext_stage,x,use_reentrant=False
        )

        tokens = self.conv_to_token(conv_out, B, V)  # (B, V, L, C)

        tokens_out, register_tokens_out = self.run_mvp_stage(tokens, B, V, info, register_tokens, hf_pe)

        trans_out = self.token_to_conv(tokens_out, H, W)  # Already normalized

        output = trans_out
        return output, register_tokens_out

    def run_mvp_stage(self, x, B, V, info, register_tokens):
        """Run MVP transformer - implemented by subclasses"""
        raise NotImplementedError




class HybridStage3(HybridStage):
    def run_mvp_stage(self, x, B, V, info, register_tokens, hf_pe=None):
        num_reg = register_tokens.shape[2] if register_tokens is not None else 0

        hf_pe_tokens = None
        if hf_pe is not None:
            BV, D, H, W = hf_pe.shape
            hf_pe_tokens = rearrange(hf_pe, '(b v) d h w -> b v (h w) d', b=B, v=V)

        if register_tokens is not None:
            x = torch.cat([register_tokens, x], dim=2)  # (B, V, R+L, D)

            if hf_pe_tokens is not None:
                zero_pe = torch.zeros(B, V, num_reg, D, device=hf_pe_tokens.device, dtype=hf_pe_tokens.dtype)
                hf_pe_tokens = torch.cat([zero_pe, hf_pe_tokens], dim=2)  # (B, V, R+L, D)

        x = rearrange(x, 'b v l d -> b (v l) d')
        if hf_pe_tokens is not None:
            hf_pe_tokens = rearrange(hf_pe_tokens, 'b v l d -> b (v l) d')

        for i, block in enumerate(self.mvp_stage):
            if i % 2 == 0:
                x = rearrange(x, 'b (v l) d -> (b v) l d', v=V)
                if hf_pe_tokens is not None:
                    hf_pe_block = rearrange(hf_pe_tokens, 'b (v l) d -> (b v) l d', v=V)
                else:
                    hf_pe_block = None
                x = torch.utils.checkpoint.checkpoint(
                    block, x, False, 3, info, hf_pe_block, use_reentrant=False)
                x = rearrange(x, '(b v) l d -> b (v l) d', v=V)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, True, 3, info, hf_pe_tokens, use_reentrant=False)
        x = rearrange(x, 'b (v l) d -> b v l d', v=V)

        if register_tokens is not None:
            register_tokens_out = x[:, :, :num_reg]  # (B, V, R, D)
            x = x[:, :, num_reg:]  # (B, V, L, D)
        else:
            register_tokens_out = None

        return x, register_tokens_out


class HybridDINOMVPModel(nn.Module):
    """
    Hybrid DINOv3-ConvNeXt + MVP Model

    Combines the strong visual features from DINOv3's ConvNeXt backbone
    with MVP's multi-view transformer reasoning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ConvNeXt configuration
        self.convnext_type = config.model.convnext_type
        if self.convnext_type == 'tiny':
            self.conv_dims = [96, 192, 384, 768]
            self.conv_depths = [3, 3, 9, 3]
        elif self.convnext_type == 'base':
            self.conv_dims = [128, 256, 512, 1024]
            self.conv_depths = [3, 3, 27, 3]
        else:
            raise ValueError(f"Unknown convnext_type: {self.convnext_type}")
        self.mvp_dims = [self.conv_dims[1], self.conv_dims[2], self.conv_dims[3]]
        self.mvp_depths = config.model.mvp_depths

        self.patch_size = config.model.patch_size
        self.num_register_tokens = config.model.get('num_register_tokens', 4)
        self.group_size = config.model.group_size
        self.head_dim = config.model.head_dim
        self.inter_multi = config.model.inter_multi
        self.qk_norm = config.model.qk_norm

        self._init_convnext_backbone()
        if hasattr(config.model, 'pretrained_path') and config.model.pretrained_path:
            self.load_pretrained_convnext(config.model.pretrained_path)
        self._init_stem_rgb()
        self._init_register_tokens()
        self._init_mvp_transformers()
        self._init_prope_attention()
        self._init_hybrid_stages()
        self._init_decoder()

        self.loss_computer = LossComputer(config)
        self.freeze_prev = config.model.get("freeze_prev", False)
        # 推理优化配置：统一使用 target_chunk_size 控制分块
        inference_cfg = config.inference if hasattr(config, "inference") else {}
        self.target_chunk_size = inference_cfg.get("target_chunk_size", None)
        if self.target_chunk_size is not None:
            print(f"[Inference Config] Target frame chunking enabled: chunk_size={self.target_chunk_size}")

        if self.freeze_prev:
            self.register_token_init.requires_grad = False
            requires_grad(self.convnext, False)

        print(f"\n{'='*60}")
        print(f"✓ HybridDINOMVPModel initialized")
        print(f"  ConvNeXt: {self.convnext_type}")
        print(f"  Dimensions: {self.mvp_dims}")
        print(f"  MVP depths: {self.mvp_depths}")
        print(f"  Register tokens: {self.num_register_tokens}")
        print(f"{'='*60}\n")

    def _init_convnext_backbone(self):
        """Initialize ConvNeXt backbone from DINOv3"""
        dinov3_path = self.config.model.get('dinov3_repo_path')
        sys.path.insert(0, dinov3_path)
        from dinov3.models.convnext import ConvNeXt

        # Create ConvNeXt to get pretrained stem weights
        self.convnext = ConvNeXt(
            in_chans=3,
            depths=self.conv_depths,
            dims=self.conv_dims,
            drop_path_rate=self.config.model.get('drop_path_rate', 0.1),
            layer_scale_init_value=self.config.model.get('layer_scale_init_value', 1e-6),
        )
        print(f"✓ ConvNeXt {self.convnext_type}: dims={self.conv_dims}")

        # Store original stem for weight initialization
        self.original_stem_conv = self.convnext.downsample_layers[0]

        # Delete norms (not used in our architecture)
        del self.convnext.norms
        del self.convnext.norm

    def _init_stem_rgb(self):
        """Initialize 3-channel RGB stem with pretrained weights"""
        # Create new RGB stem with pretrained weights
        self.convnext.stem_rgb = RGBStem(
            pretrained_stem=self.original_stem_conv
        )
        del self.original_stem_conv
        # import pdb;pdb.set_trace()
        self.convnext.downsample_layers[0] = nn.Identity()  # stem
        # Replace ConvNeXt's original stem with our RGB stem
        # We'll bypass downsample_layers[0] in forward pass
        print(f"✓ RGBStem: 3 channels (RGB) → {self.conv_dims[0]}")

    def _init_register_tokens(self):
        """Initialize learnable register tokens (shared across all stages)"""
        if self.num_register_tokens > 0:
            # Single register token initialized with dim1, will be resized for different stages
            self.register_token_init = nn.Parameter(
                torch.zeros(1, 1, self.num_register_tokens, self.mvp_dims[-1]))
            nn.init.normal_(self.register_token_init, mean=0.0, std=0.02)

            print(f"✓ Register Tokens: {self.num_register_tokens} tokens (resized across stages)")
        else:
            self.register_token_init = None
            self.resize_register1 = None
            self.resize_register2 = None

    def _init_mvp_transformers(self):
        """Initialize MVP transformer stages - Stage 3 only with HFPE"""

        self.stage3 = nn.ModuleList([
            TransformerBlockWithHFPE(self.mvp_dims[2], False, self.head_dim,
                           self.inter_multi, self.qk_norm)
            for _ in range(self.mvp_depths[2])
        ])
        self.stage3.apply(_init_weights)

        self.hfpe_generator = FPA(
            in_channels=self.conv_dims[0],  # e.g., 96 for tiny, 128 for base
            out_channels=self.mvp_dims[2],  # e.g., 768 for tiny, 1024 for base
            wavelet='haar'
        )

        print(f"✓ MVP Transformers: stage3 dim={self.mvp_dims[2]}")
        print(f"  stage3_nlayer={self.mvp_depths[2]}")
        print(f"✓ High-Freq PE: stage0({self.conv_dims[0]}) -> stage3({self.mvp_dims[2]})")

    def _init_prope_attention(self):
        """Initialize ProPe attention modules - Stage 3 only"""
        H, W = self.config.training.image_height, self.config.training.image_width

        # Only Stage 3 attention module
        self.prope_attn3 = PropeDotProductAttention(
            head_dim=self.head_dim, patches_x=W // 32, patches_y=H // 32,
            image_width=W, image_height=H, num_register_tokens=self.num_register_tokens)

        print(f"✓ ProPe Attention module (stage3)")

    def _init_hybrid_stages(self):
        """Initialize hybrid stages - only Stage 3"""
        # Stage 1 and 2: ConvNeXt only
        # Stage 3: ConvNeXt + MVP Transformer (direct addition, no alpha)
        self.hybrid_stage3 = HybridStage3(
            self.convnext.stages[3], self.stage3, self.mvp_dims[2])

        print(f"✓ Stage 3: ConvNeXt + MVP Transformer (direct addition, no alpha)")

    def _init_decoder(self):
        """Initialize channel alignment modules, DPT head and Gaussian decoder
        """
        # Channel alignment modules: (C+9) -> C for 4 stages
        self.align0 = ChannelAlignmentModule(self.conv_dims[0] + 9, self.conv_dims[0])
        self.align1 = ChannelAlignmentModule(self.mvp_dims[0] + 9, self.mvp_dims[0])
        self.align2 = ChannelAlignmentModule(self.mvp_dims[1] + 9, self.mvp_dims[1])
        self.align3 = ChannelAlignmentModule(self.mvp_dims[2] + 9, self.mvp_dims[2])


        dpt_dims = [self.conv_dims[0]] + self.mvp_dims  # [96, 192, 384, 768] for tiny
        self.dpt_head = DPTHead(
            dim_in=dpt_dims,
            features=self.mvp_dims[2],
            out_channels=dpt_dims)

        color_dim = 3 * (self.config.model.gaussians.sh_degree + 1) ** 2
        opacity_dim = 1 * (self.config.model.gaussians.opacity_degree + 1) ** 2
        gaussian_output_dim = 3 + color_dim + 3 + 4 + opacity_dim

        self.gaussian_decoder = nn.Sequential(
            nn.LayerNorm(self.mvp_dims[2], bias=False),
            nn.Linear(self.mvp_dims[2], ((self.patch_size//2) ** 2) * gaussian_output_dim, bias=False))

        print(f"✓ Channel Alignment: [{self.conv_dims[0]+9}->{self.conv_dims[0]}, "
              f"{self.mvp_dims[0]+9}->{self.mvp_dims[0]}, {self.mvp_dims[1]+9}->{self.mvp_dims[1]}, "
              f"{self.mvp_dims[2]+9}->{self.mvp_dims[2]}]")
        print(f"✓ Decoder: DPT input dims={dpt_dims}, gaussian_dim={gaussian_output_dim}")

    def load_pretrained_convnext(self, pretrain_path):
        """Load pretrained ConvNeXt weights from DINOv3"""
        print(f"\nLoading pretrained ConvNeXt: {pretrain_path}")

        if not os.path.exists(pretrain_path):
            print(f"⚠ Warning: File not found")
            return

        checkpoint = torch.load(pretrain_path, map_location='cpu')
        pretrain_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        model_dict = self.convnext.state_dict()

        matched_dict = {k: v for k, v in pretrain_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(matched_dict)
        missing_keys, unexpected_keys = self.convnext.load_state_dict(model_dict, strict=False)
        print('missing keys = ',missing_keys)
        print('unexpected_keys = ',unexpected_keys)
        print(f"✓ Loaded {len(matched_dict)}/{len(model_dict)} parameters")
    def train(self, mode=True):
        super().train(mode)
        self.loss_computer.eval()
        return self

    def forward(self, input_data_dict, target_data_dict):
        """Forward pass"""
        # Prepare data
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            B, V, _, H, W = input_data_dict["image"].size()
            print(f"B={B}, V={V}, H={H}, W={W}")
            T = target_data_dict["image"].size(1)

            i_fxfycxcy = input_data_dict["fxfycxcy"]
            i_c2w = input_data_dict["c2w"]
            t_fxfycxcy = target_data_dict["fxfycxcy"]
            t_c2w = target_data_dict["c2w"]
            # import pdb;pdb.set_trace()
            # Compute ray features at original resolution
            ray_o, ray_d = compute_plucmap(i_fxfycxcy, i_c2w, H, W)
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)

            Ks = torch.eye(3, dtype=i_c2w.dtype, device=i_c2w.device).unsqueeze(0).unsqueeze(0)
            Ks = Ks.repeat(B, V, 1, 1).clone()
            Ks[:, :, 0, 0] = i_fxfycxcy[:, :, 0]
            Ks[:, :, 1, 1] = i_fxfycxcy[:, :, 1]
            Ks[:, :, 0, 2] = i_fxfycxcy[:, :, 2]
            Ks[:, :, 1, 2] = i_fxfycxcy[:, :, 3]
            i_w2c = torch.inverse(i_c2w)

        # Use RGB stem (3 channels only)
        x = self.convnext.stem_rgb(input_data_dict["image"])

        # Initialize register tokens
        if self.num_register_tokens > 0:
            register_tokens = self.register_token_init.repeat(B, V, 1, 1)  # (B, V, R, dim1)
        else:
            register_tokens = None

        # Stage 0 (ConvNeXt only, no transformer)
        x = self.convnext.stages[0](x)
        # x = torch.utils.checkpoint.checkpoint(
        #     self.convnext.stages[0],x,use_reentrant=False
        # )
        # Save stage0 features for high-frequency PE generation
        feat_0_for_hfpe = x  # (B*V, conv_dims[0], H/4, W/4)

        # Compute ray features at stage0 resolution (H/4, W/4)
        _, _, h0, w0 = x.shape
        ray_feat0 = downsample_ray_features(ray_o, ray_d, o_cross_d, h0, w0)
        # Concatenate: x + ray_feat0 -> (B*V, C+9, H/4, W/4)
        feat0 = torch.cat([x, ray_feat0], dim=1)

        # Stage 1: ConvNeXt only, no MVP Transformer
        x = self.convnext.downsample_layers[1](x)
        x = self.convnext.stages[1](x)
        # x = torch.utils.checkpoint.checkpoint(
        #     self.convnext.stages[1],x,use_reentrant=False
        # )
        # Compute ray features at stage1 resolution (H/8, W/8)
        _, _, h1, w1 = x.shape
        ray_feat1 = downsample_ray_features(ray_o, ray_d, o_cross_d, h1, w1)
        # Concatenate: x + ray_feat1 -> (B*V, C+9, H/8, W/8)
        feat1 = torch.cat([x, ray_feat1], dim=1)

        # Resize register tokens for stage 2
        # if register_tokens is not None:
        #     register_tokens = self.resize_register1(register_tokens)  # (B, V, R, dim2)

        # Stage 2: ConvNeXt only, no MVP Transformer
        x = self.convnext.downsample_layers[2](x)
        # x = self.convnext.stages[2](x)
        x = torch.utils.checkpoint.checkpoint(
            self.convnext.stages[2],x,use_reentrant=False
        )
        # Compute ray features at stage2 resolution (H/16, W/16)
        _, _, h2, w2 = x.shape
        ray_feat2 = downsample_ray_features(ray_o, ray_d, o_cross_d, h2, w2)
        # Concatenate: x + ray_feat2 -> (B*V, C+9, H/16, W/16)
        feat2 = torch.cat([x, ray_feat2], dim=1)

        # Resize register tokens for stage 3
        # if register_tokens is not None:
        #     register_tokens = self.resize_register2(register_tokens)  # (B, V, R, dim3)

        # Stage 3: ConvNeXt + MVP Transformer with HFPE
        x = self.convnext.downsample_layers[3](x)

        # Generate high-frequency positional encoding from stage0 features
        hf_pe = self.hfpe_generator(feat_0_for_hfpe)  # (B*V, mvp_dims[2], H/32, W/32)

        info3 = {'w2c': i_w2c, 'Ks': Ks, 'attn3': self.prope_attn3}
        x, register_tokens = self.hybrid_stage3(x, B, V, H//32, W//32, info3, register_tokens, hf_pe)
        # Compute ray features at stage3 resolution (H/32, W/32)
        _, _, h3, w3 = x.shape
        ray_feat3 = downsample_ray_features(ray_o, ray_d, o_cross_d, h3, w3)
        # Concatenate: x + ray_feat3 -> (B*V, C+9, H/32, W/32)
        feat3 = torch.cat([x, ray_feat3], dim=1)

        # Channel alignment: (C+9) -> C for each stage before DPT
        feat0_aligned = self.align0(feat0)  # (B*V, C0, H/4, W/4)
        feat1_aligned = self.align1(feat1)  # (B*V, C1, H/8, W/8)
        feat2_aligned = self.align2(feat2)  # (B*V, C2, H/16, W/16)
        feat3_aligned = self.align3(feat3)  # (B*V, C3, H/32, W/32)

        # Convert aligned features to tokens for DPT Head
        feat0_tokens = rearrange(feat0_aligned, '(b v) c h w -> (b v) (h w) c', b=B, v=V)
        feat1_tokens = rearrange(feat1_aligned, '(b v) c h w -> (b v) (h w) c', b=B, v=V)
        feat2_tokens = rearrange(feat2_aligned, '(b v) c h w -> (b v) (h w) c', b=B, v=V)
        feat3_tokens = rearrange(feat3_aligned, '(b v) c h w -> (b v) (h w) c', b=B, v=V)

        # DPT Head (receives 4 aligned features with ray information fused)
        # output_tokens = self.dpt_head([ feat0_tokens,feat1_tokens, feat2_tokens, feat3_tokens], [H, W], self.patch_size)
        output_tokens = torch.utils.checkpoint.checkpoint(
            self.dpt_head,[ feat0_tokens,feat1_tokens, feat2_tokens, feat3_tokens], [H, W], self.patch_size,use_reentrant=False
        )
        
        output_tokens = rearrange(output_tokens, '(b v) l d -> b (v l) d', b=B, v=V)

        # Gaussian Decoder
        gaussians = self.gaussian_decoder(output_tokens)
        color_dim = 3 * (self.config.model.gaussians.sh_degree + 1) ** 2
        opacity_dim = 1 * (self.config.model.gaussians.opacity_degree + 1) ** 2
        # import pdb;pdb.set_trace()
        # return 
        gaussians = rearrange(
            gaussians, 'b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d', v=V,
            hh=H // (self.patch_size//2 ), ww=W // (self.patch_size//2),
            ph=self.patch_size//2, pw=self.patch_size//2)

        xyz, feature, scale, rotation, opacity = torch.split(
            gaussians, [3, color_dim, 3, 4, opacity_dim], dim=-1)

        xyz, feature, scale, rotation, opacity = xyz.float(), feature.float(), scale.float(), rotation.float(), opacity.float()

        with torch.autocast(device_type="cuda", enabled=False):
            rayo_gs, rayd_gs = compute_rays(i_fxfycxcy, i_c2w, H, W)
            scale = (scale + self.config.model.gaussians.scale_bias).clamp(max=self.config.model.gaussians.scale_max)
            opacity[..., 0] = opacity[..., 0] + self.config.model.gaussians.opacity_bias

            feature = rearrange(feature, 'b n (c d) -> b n d c', c=3).contiguous()
            opacity = rearrange(opacity, 'b n (c d) -> b n d c', c=1).contiguous()

            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.config.model.gaussians.max_dist
            xyz = dist * rayd_gs + rayo_gs

        # 合并 opacity 预计算与渲染到同一 chunk 循环：
        # 每次只保留 1 帧的 opacity_precompute，渲染完立即释放，避免全量 (B,T,N,1) 驻留显存。
        with torch.autocast(device_type="cuda", enabled=False):
            if self.target_chunk_size is not None and T > self.target_chunk_size:
                renderings_chunks = []
                for chunk_start in range(0, T, self.target_chunk_size):
                    chunk_end = min(chunk_start + self.target_chunk_size, T)
                    t_c2w_chunk = t_c2w[:, chunk_start:chunk_end]
                    dirs_chunk = xyz[:, None, :, :] - t_c2w_chunk[..., :3, 3][..., None, :]
                    opacity_broad_chunk = torch.broadcast_to(
                        opacity[..., None, :, :, :],
                        (B, chunk_end - chunk_start, opacity.shape[1], -1, 1),
                    )
                    opacity_chunk = _spherical_harmonics(
                        self.config.model.gaussians.opacity_degree,
                        dirs_chunk,
                        opacity_broad_chunk,
                    )
                    renderings_chunk = GaussianRenderer.apply(
                        xyz, feature, scale, rotation, opacity_chunk,
                        t_c2w_chunk, t_fxfycxcy[:, chunk_start:chunk_end],
                        W, H,
                        self.config.model.gaussians.sh_degree,
                        self.config.model.gaussians.near_plane,
                        self.config.model.gaussians.far_plane,
                    )
                    renderings_chunks.append(renderings_chunk)
                    del opacity_chunk, dirs_chunk, opacity_broad_chunk
                renderings = torch.cat(renderings_chunks, dim=1)
            else:
                dirs = xyz[:, None, :, :] - t_c2w[..., :3, 3][..., None, :]
                opacity_broad = torch.broadcast_to(opacity[..., None, :, :, :], (B, T, opacity.shape[1], -1, 1))
                opacity_precompute = _spherical_harmonics(self.config.model.gaussians.opacity_degree, dirs, opacity_broad)
                renderings = GaussianRenderer.apply(
                    xyz, feature, scale, rotation, opacity_precompute, t_c2w, t_fxfycxcy,
                    W, H, self.config.model.gaussians.sh_degree,
                    self.config.model.gaussians.near_plane, self.config.model.gaussians.far_plane)

        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous()

        # 推理模式不计算训练损失
        loss_metrics = None
        return edict(input=input_data_dict, target=target_data_dict,
                    loss_metrics=loss_metrics, render=renderings)
    def set_inference_config(self, target_chunk_size=None):
        """
        设置推理时的内存优化配置

        Args:
            target_chunk_size (int, optional): 统一的目标帧分块大小，
                同时用于 opacity 预计算和渲染分块
        """
        self.target_chunk_size = target_chunk_size
        if target_chunk_size is not None:
            print(f"[Inference Config] Target frame chunking enabled: chunk_size={target_chunk_size}")

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        msg = self.load_state_dict(checkpoint["ema"], strict=False)
        print("Missing keys:", msg.missing_keys)

        print("Unexpected keys:", msg.unexpected_keys)
        
        return 0