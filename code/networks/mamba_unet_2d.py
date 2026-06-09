import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleMambaBlock(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.in_proj = nn.Conv2d(dim, dim * expansion, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim * expansion, dim * expansion, kernel_size=3, padding=1, groups=dim * expansion, bias=False)
        self.gate = nn.Conv2d(dim * expansion, dim * expansion, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(dim * expansion, dim, kernel_size=1, bias=False)
        self.ffn = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim * expansion, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = self.dwconv(x)
        x = torch.tanh(self.gate(x)) * x
        x = self.out_proj(x)
        x = residual + x
        return x + self.ffn(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        layers = [ConvNormAct(in_channels, out_channels)]
        for _ in range(depth):
            layers.append(SimpleMambaBlock(out_channels))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, depth):
        super().__init__()
        self.proj = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False)
        blocks = []
        for _ in range(max(depth, 1)):
            blocks.append(SimpleMambaBlock(out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.blocks(x)


class MambaUNet2D(nn.Module):
    def __init__(
        self,
        in_chns=1,
        class_num=4,
        img_size=256,
        base_channels=32,
        depths=(1, 1, 1, 1),
        dims=(32, 64, 128, 256),
        return_feature_stage="decoder_low",
        mamba_variant="vmunet",
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.class_num = class_num
        self.return_feature_stage = return_feature_stage
        self.mamba_variant = mamba_variant

        self.enc0 = EncoderStage(in_chns, dims[0], depths[0])
        self.enc1 = EncoderStage(dims[0], dims[1], depths[1])
        self.enc2 = EncoderStage(dims[1], dims[2], depths[2])
        self.enc3 = EncoderStage(dims[2], dims[3], depths[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvNormAct(dims[3], dims[3]),
            SimpleMambaBlock(dims[3]),
        )

        self.dec3 = DecoderStage(dims[3], dims[3], dims[2], depth=1)
        self.dec2 = DecoderStage(dims[2], dims[2], dims[1], depth=1)
        self.dec1 = DecoderStage(dims[1], dims[1], dims[0], depth=1)
        self.dec0 = DecoderStage(dims[0], dims[0], dims[0], depth=1)
        self.head = nn.Conv2d(dims[0], class_num, kernel_size=1)

        self.feature_channels = dims[2]

    def forward(self, x, return_features=True):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)
        logits = self.head(d0)

        if not return_features:
            return logits

        if self.return_feature_stage == "decoder_low":
            feature = d3
        elif self.return_feature_stage == "decoder_mid":
            feature = d2
        elif self.return_feature_stage == "decoder_last":
            feature = d0
        elif self.return_feature_stage == "bottleneck":
            feature = x4
        else:
            raise ValueError("Unsupported return_feature_stage: {}".format(self.return_feature_stage))
        return logits, feature
