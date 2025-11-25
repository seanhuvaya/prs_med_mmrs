import torch
import torch.nn as nn
from timm.layers import DropPath as TimmDropPath, to_2tuple, trunc_normal_
from torchvision import transforms

from utils import logging

logger = logging.get_logger(__name__)


# ==========================
# Internal TinyViT components
# (extracted minimal subset from tinysam TinyViT implementation)
# ==========================

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1), w.size(0), w.size(2), w.size(3), bias=True)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)

    def __repr__(self):
        return f'DropPath(drop_prob={self.drop_prob})'


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        # resolution: (H, W)
        # Use attribute name `seq` to match TinySAM/TinyViT checkpoint keys (patch_embed.seq.*)
        self.seq = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim // 2, ks=3, stride=2, pad=1),
            activation(),
            Conv2d_BN(embed_dim // 2, embed_dim, ks=3, stride=2, pad=1),
        )
        self.resolution = (resolution[0] // 4, resolution[1] // 4)

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        hidden = int(in_chans * expand_ratio)
        # conv1: pointwise 1x1
        self.conv1 = Conv2d_BN(in_chans, hidden, ks=1, stride=1, pad=0)
        self.act1 = activation()
        # conv2: depthwise 3x3
        self.conv2 = Conv2d_BN(hidden, hidden, ks=3, stride=1, pad=1, groups=hidden)
        self.act2 = activation()
        # conv3: pointwise 1x1 with BN weight init to 0.0 (to ease residual learning)
        self.conv3 = Conv2d_BN(hidden, out_chans, ks=1, stride=1, pad=0, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x = x + shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()
        # Align naming and structure with TinySAM/TinyViT checkpoints:
        # conv1: 1x1 pointwise
        # conv2: depthwise 3x3 with stride logic
        # conv3: 1x1 pointwise
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()

        # conv1
        self.conv1 = Conv2d_BN(dim, out_dim, ks=1, stride=1, pad=0)

        # stride logic copied from TinySAM implementation
        stride_c = 2
        if out_dim in (320, 448, 576):
            stride_c = 1

        # conv2 depthwise
        self.conv2 = Conv2d_BN(out_dim, out_dim, ks=3, stride=stride_c, pad=1, groups=out_dim)

        # conv3
        self.conv3 = Conv2d_BN(out_dim, out_dim, ks=1, stride=1, pad=0)

    def forward(self, x):
        # Keep NCHW flow, unlike the original TinySAM which flattens to tokens here.
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, activation, drop_path=0., downsample=None, use_checkpoint=False, out_dim=None, conv_expand_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, expand_ratio=conv_expand_ratio, activation=activation, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Add LayerNorm to match TinySAM checkpoints: mlp.norm.*
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Normalize tokens before feed-forward, aligning with TinySAM
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim=None, num_heads=8, attn_ratio=1, resolution=(14, 14), window_size=7):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # Some TinySAM checkpoints include a LayerNorm inside attention as `attn.norm.*`
        self.norm = nn.LayerNorm(dim)
        # TinySAM stores `attention_biases` per window; keep the parameter for checkpoint compatibility.
        ws = window_size
        # Match checkpoint shape: [num_heads, window_size * window_size]
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, ws * ws))
        self.resolution = resolution
        self.window_size = window_size

    def train(self, mode=True):
        super().train(mode)
        return self

    def forward(self, x):
        B, N, C = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # We do not apply attention_biases unless N matches window_size*window_size; parameter remains for loading.
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., drop=0., drop_path=0., local_conv_size=3, activation=nn.GELU):
        super().__init__()
        self.attn = Attention(
            dim,
            key_dim=dim // num_heads,
            num_heads=num_heads,
            attn_ratio=1,
            resolution=input_resolution,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop)
        # Add the TinySAM-local depthwise conv between attention and MLP.
        # Must be named `local_conv` to match checkpoint keys: layers.*.blocks.*.local_conv.*
        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Tokenize for attention
        x_tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_tokens = x_tokens + self.drop_path(self.attn(x_tokens))

        # Bring back to NCHW to apply local depthwise convolution
        x_nchw = x_tokens.transpose(1, 2).reshape(B, C, H, W)
        x_nchw = self.local_conv(x_nchw)

        # Back to tokens for MLP
        x_tokens = x_nchw.flatten(2).transpose(1, 2)
        x_tokens = x_tokens + self.drop_path(self.mlp(x_tokens))

        # Return to NCHW
        x_out = x_tokens.transpose(1, 2).reshape(B, C, H, W)
        return x_out

    def extra_repr(self):
        return f"input_resolution=({self.attn.resolution[0]}, {self.attn.resolution[1]})"


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., drop=0., drop_path=0., downsample=None, use_checkpoint=False, local_conv_size=3, activation=nn.GELU, out_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TinyViTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                local_conv_size=local_conv_size,
                activation=activation,
            ) for _ in range(depth)
        ])
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return f"input_resolution=?"


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TinyViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[96, 192, 384, 768],
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0):
        super().__init__()

        self.img_size = img_size
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0],
            resolution=to_2tuple(img_size), activation=activation)

        self.layers = nn.ModuleList()
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0
        for i_layer in range(len(embed_dims)):
            dim = embed_dims[i_layer]
            input_resolution = (img_size // (4 * (2 ** i_layer)), img_size // (4 * (2 ** i_layer)))
            depth = depths[i_layer]
            drop_path = dpr[cur:cur + depth]
            cur += depth
            out_dim = embed_dims[i_layer + 1] if i_layer < len(embed_dims) - 1 else None
            if i_layer == 0:
                downsample = PatchMerging(input_resolution, dim, embed_dims[i_layer + 1], activation=activation)
            else:
                downsample = PatchMerging(input_resolution, dim, embed_dims[i_layer + 1] if out_dim is not None else dim, activation=activation) if i_layer < len(embed_dims) - 1 else None

            kwargs = dict(dim=dim, input_resolution=input_resolution, depth=depth,
                          drop_path=drop_path[0] if isinstance(drop_path, list) and len(drop_path) > 0 else 0.,
                          downsample=downsample, use_checkpoint=use_checkpoint, out_dim=out_dim,
                          activation=activation)
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay
        depth = sum([len(getattr(layer, 'blocks', [])) for layer in self.layers])
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0] if lr_scales else 1.0))
        i = 0
        for layer in self.layers:
            for block in getattr(layer, 'blocks', []):
                block.apply(lambda x: _set_lr_scale(x, lr_scales[min(i, len(lr_scales)-1)]))
                i += 1
            if getattr(layer, 'downsample', None) is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[min(i - 1, len(lr_scales)-1)]))
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1] if lr_scales else 1.0))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), getattr(p, 'param_name', 'param')

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        # At this point x is NCHW feature map at 1/16 resolution (64x64 for 1024 inputs)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

class TinySAMVisionBackbone(nn.Module):
    """
    Extracts dense, pixel-level features from medical images using TinyViT-based TinySAM encoder.
    """

    def __init__(self, checkpoint_path: str, device: str, image_size: int = 1024):
        super().__init__()

        # Persist config
        self.device = torch.device(device)
        self.image_size = image_size

        logger.info(f"Loading TinySAM checkpoint from {checkpoint_path}")

        # Instantiate only the TinyViT image encoder with TinySAM's configuration
        encoder = TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )

        # Load weights, extracting only the image_encoder subset if the checkpoint
        # contains a full SAM state_dict.
        try:
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and any(k.startswith("image_encoder.") for k in state.keys()):
                # Filter and strip the prefix
                encoder_state = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
                missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
                if missing:
                    logger.warning(f"Missing image encoder keys: {len(missing)} (showing 5): {missing[:5]}")
                if unexpected:
                    logger.warning(f"Unexpected image encoder keys: {len(unexpected)} (showing 5): {unexpected[:5]}")
            else:
                # Assume the checkpoint is directly for TinyViT
                encoder.load_state_dict(state, strict=False)
        except Exception as e:
            logger.error(f"Failed to load TinySAM/TinyViT weights from {checkpoint_path}: {e}")
            raise

        # Move to device
        self.encoder = encoder.to(self.device)

        # Keep encoder unfrozen for medical fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Define preprocessing consistent with SAM training (ImageNet stats)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack([self.transform(img) for img in x])
        elif isinstance(x, torch.Tensor):
            # If tensor is already ImageNet-normalized (can have values outside [0,1]),
            # assume it's ready. Otherwise, if values are in [0, 255] range, normalize.
            # Simple heuristic: if max > 2.0, likely unnormalized (divide by 255)
            # If already normalized, values typically in range roughly [-2, 2]
            if x.max() > 2.0:
                # Likely unnormalized [0, 255] range
                x = x / 255.0
                # Apply ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std
            # If already normalized (max <= 2.0), assume it's ImageNet-normalized and use as-is

        x = x.to(self.device)

        with torch.set_grad_enabled(self.training):
            z_image = self.encoder(x)

        if z_image.shape[-2:] == (64, 64):
            # 64x64 → 16x16 (4x downsampling)
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))
        elif z_image.shape[-2:] != (16, 16):
            # Fallback: ensure we get 16x16
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))

        return z_image


if __name__ == "__main__":
    device = torch.device("mps")
    model = TinySAMVisionBackbone(checkpoint_path="weights/tinysam_42.3.pth", device="mps")
    dummy = torch.randn(1, 3, 1024, 1024)
    features = model(dummy)
    print("Output shape:", features.shape)
