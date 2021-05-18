import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_pad,stride=2):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(kernel_size_pad, stride=stride), DoubleConv(in_channels, out_channels, 3))

    def forward(self, x):
        out = self.block(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_up,padding=0,stride=2, out_pad=0, upsample=None):
        super().__init__()
        if upsample:
            self.up_s = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.up_s = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size_up, stride=stride, padding=padding,
                                           output_padding=out_pad)

        self.convT = DoubleConv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        out = self.up_s(x1)
        out = self.convT(torch.cat((x2, out), dim=1))
        return out


class Unet(nn.Module):
    def __init__(self, n_classes, upsample):
        super().__init__()
        self.n_classes = n_classes

        self.in1 = DoubleConv(14, 32, 3)
        self.down1 = Down(32, 64, 3)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        factor = 2 if upsample else 1
        self.down4 = Down(256, 512 // factor, 3)
        self.up1 = Up(512, 256 // factor, 3, upsample=upsample,stride=2,out_pad=0)
        self.up2 = Up(256, 128 // factor, 3, upsample=upsample)
        self.up3 = Up(128, 64 // factor, 3, upsample=upsample,out_pad=1)
        self.up4 = Up(64, 32, 3, upsample=upsample)
        self.conv = nn.Conv3d(32, self.n_classes, 1)

    def forward(self, x):
        x1 = self.in1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.conv(x)
        return logits


