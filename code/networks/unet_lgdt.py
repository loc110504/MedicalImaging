import torch
import torch.nn as nn

from networks.unet import ConvBlock, Encoder, UpBlock


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        branches = []
        for r in rates:
            if r == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=r,
                    dilation=r,
                    bias=False,
                )
            branches.append(
                nn.Sequential(
                    conv,
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class DecoderLGDT(nn.Module):
    def __init__(self, ft_chns, class_num, refine_dropout=0.0):
        super().__init__()
        self.up1 = UpBlock(ft_chns[4], ft_chns[3], ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(ft_chns[3], ft_chns[2], ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(ft_chns[2], ft_chns[1], ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(ft_chns[1], ft_chns[0], ft_chns[0], dropout_p=0.0)
        self.refine = ConvBlock(ft_chns[0], ft_chns[0], dropout_p=refine_dropout)
        self.out_conv = nn.Conv2d(ft_chns[0], class_num, kernel_size=1)

    def forward(self, feature, bottleneck=None):
        x0, x1, x2, x3, x4 = feature
        if bottleneck is not None:
            x4 = bottleneck

        x = self.up1(x4, x3)
        high = x
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        low = self.refine(x)
        logits = self.out_conv(low)
        return logits, high, low


class UNet_LGDT(nn.Module):
    def __init__(
        self,
        in_chns=1,
        class_num=4,
        base_chns=16,
        return_all_default=False,
        use_aspp=True,
    ):
        super().__init__()
        params = {
            "in_chns": in_chns,
            "feature_chns": [base_chns, base_chns * 2, base_chns * 4, base_chns * 8, base_chns * 16],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "bilinear": False,
            "acti_func": "relu",
        }
        self.return_all_default = return_all_default
        self.encoder = Encoder(params)
        self.decoder_s = DecoderLGDT(params["feature_chns"], class_num)
        self.decoder_l = DecoderLGDT(params["feature_chns"], class_num)
        self.decoder_g = DecoderLGDT(params["feature_chns"], class_num)
        self.context = ASPP(params["feature_chns"][4], params["feature_chns"][4]) if use_aspp else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_all=None):
        if return_all is None:
            return_all = self.return_all_default

        feats = self.encoder(x)
        logits_s, high_s, low_s = self.decoder_s(feats)
        logits_l, high_l, low_l = self.decoder_l(feats)

        bottleneck = self.context(feats[-1])
        logits_g, high_g, low_g = self.decoder_g(feats, bottleneck=bottleneck)

        if return_all:
            return {
                "logits_s": logits_s,
                "logits_l": logits_l,
                "logits_g": logits_g,
                "low_s": low_s,
                "low_l": low_l,
                "low_g": low_g,
                "high_s": high_s,
                "high_l": high_l,
                "high_g": high_g,
            }

        return logits_s, high_s, low_s
