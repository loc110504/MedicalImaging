import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type="kaiming", gain=0.02):
    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise NotImplementedError("Unsupported init_type: {}".format(init_type))
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(module.weight.data, 1.0, gain)
            init.constant_(module.bias.data, 0.0)

    net.apply(init_func)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class XNetv2(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=64):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.m_conv1 = ConvBlock(in_channels, channels[0])
        self.m_conv2 = ConvBlock(channels[0], channels[1])
        self.m_conv3 = ConvBlock(channels[1], channels[2])
        self.m_conv4 = ConvBlock(channels[2], channels[3])
        self.m_conv5 = ConvBlock(channels[3], channels[4])
        self.m_up5 = UpConv(channels[4], channels[3])
        self.m_up_conv5 = ConvBlock(channels[4], channels[3])
        self.m_up4 = UpConv(channels[3], channels[2])
        self.m_up_conv4 = ConvBlock(channels[3], channels[2])
        self.m_up3 = UpConv(channels[2], channels[1])
        self.m_up_conv3 = ConvBlock(channels[2], channels[1])
        self.m_up2 = UpConv(channels[1], channels[0])
        self.m_up_conv2 = ConvBlock(channels[1], channels[0])
        self.m_head = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.l_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l_conv1 = ConvBlock(in_channels, channels[0])
        self.l_conv2 = ConvBlock(channels[0], channels[1])
        self.l_conv3 = ConvBlock(channels[1], channels[2])
        self.l_conv4 = ConvBlock(channels[2], channels[3])
        self.l_conv5 = ConvBlock(channels[3], channels[4])
        self.l_up5 = UpConv(channels[4], channels[3])
        self.l_up_conv5 = ConvBlock(channels[4], channels[3])
        self.l_up4 = UpConv(channels[3], channels[2])
        self.l_up_conv4 = ConvBlock(channels[3], channels[2])
        self.l_up3 = UpConv(channels[2], channels[1])
        self.l_up_conv3 = ConvBlock(channels[2], channels[1])
        self.l_up2 = UpConv(channels[1], channels[0])
        self.l_up_conv2 = ConvBlock(channels[1], channels[0])
        self.l_head = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.h_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.h_conv1 = ConvBlock(in_channels, channels[0])
        self.h_conv2 = ConvBlock(channels[0], channels[1])
        self.h_conv3 = ConvBlock(channels[1], channels[2])
        self.h_conv4 = ConvBlock(channels[2], channels[3])
        self.h_conv5 = ConvBlock(channels[3], channels[4])
        self.h_up5 = UpConv(channels[4], channels[3])
        self.h_up_conv5 = ConvBlock(channels[4], channels[3])
        self.h_up4 = UpConv(channels[3], channels[2])
        self.h_up_conv4 = ConvBlock(channels[3], channels[2])
        self.h_up3 = UpConv(channels[2], channels[1])
        self.h_up_conv3 = ConvBlock(channels[2], channels[1])
        self.h_up2 = UpConv(channels[1], channels[0])
        self.h_up_conv2 = ConvBlock(channels[1], channels[0])
        self.h_head = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.m_h_conv1 = ConvBlock(channels[0] * 2, channels[0])
        self.m_h_conv2 = ConvBlock(channels[1] * 2, channels[1])
        self.m_l_conv3 = ConvBlock(channels[2] * 2, channels[2])
        self.m_l_conv4 = ConvBlock(channels[3] * 2, channels[3])

        init_weights(self, "kaiming")

    def forward(self, x_main, x_low, x_high):
        m_x1 = self.m_conv1(x_main)
        m_x2 = self.m_conv2(self.m_pool(m_x1))
        m_x3 = self.m_conv3(self.m_pool(m_x2))
        m_x4 = self.m_conv4(self.m_pool(m_x3))
        m_x5 = self.m_conv5(self.m_pool(m_x4))

        l_x1 = self.l_conv1(x_low)
        l_x2 = self.l_conv2(self.l_pool(l_x1))
        l_x3 = self.l_conv3(self.l_pool(l_x2))
        l_x4 = self.l_conv4(self.l_pool(l_x3))
        l_x5 = self.l_conv5(self.l_pool(l_x4))

        h_x1 = self.h_conv1(x_high)
        h_x2 = self.h_conv2(self.h_pool(h_x1))
        h_x3 = self.h_conv3(self.h_pool(h_x2))
        h_x4 = self.h_conv4(self.h_pool(h_x3))
        h_x5 = self.h_conv5(self.h_pool(h_x4))

        m_h_x1 = self.m_h_conv1(torch.cat((m_x1, h_x1), dim=1))
        m_h_x2 = self.m_h_conv2(torch.cat((m_x2, h_x2), dim=1))
        m_l_x3 = self.m_l_conv3(torch.cat((m_x3, l_x3), dim=1))
        m_l_x4 = self.m_l_conv4(torch.cat((m_x4, l_x4), dim=1))

        m_d5 = self.m_up_conv5(torch.cat((m_l_x4, self.m_up5(m_x5)), dim=1))
        m_d4 = self.m_up_conv4(torch.cat((m_l_x3, self.m_up4(m_d5)), dim=1))
        m_d3 = self.m_up_conv3(torch.cat((m_h_x2, self.m_up3(m_d4)), dim=1))
        m_d2 = self.m_up_conv2(torch.cat((m_h_x1, self.m_up2(m_d3)), dim=1))
        main_logits = self.m_head(m_d2)

        l_d5 = self.l_up_conv5(torch.cat((m_l_x4, self.l_up5(l_x5)), dim=1))
        l_d4 = self.l_up_conv4(torch.cat((m_l_x3, self.l_up4(l_d5)), dim=1))
        l_d3 = self.l_up_conv3(torch.cat((l_x2, self.l_up3(l_d4)), dim=1))
        l_d2 = self.l_up_conv2(torch.cat((l_x1, self.l_up2(l_d3)), dim=1))
        low_logits = self.l_head(l_d2)

        h_d5 = self.h_up_conv5(torch.cat((h_x4, self.h_up5(h_x5)), dim=1))
        h_d4 = self.h_up_conv4(torch.cat((h_x3, self.h_up4(h_d5)), dim=1))
        h_d3 = self.h_up_conv3(torch.cat((m_h_x2, self.h_up3(h_d4)), dim=1))
        h_d2 = self.h_up_conv2(torch.cat((m_h_x1, self.h_up2(h_d3)), dim=1))
        high_logits = self.h_head(h_d2)

        return main_logits, low_logits, high_logits
