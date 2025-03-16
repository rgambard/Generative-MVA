import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class ResidualUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(ResidualUNet, self).__init__()

        # Encoder
        self.enc1 = ResidualBlock(input_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        # Decoder
        self.dec3 = ResidualBlock( 512+512, 256)
        self.dec2 = ResidualBlock(256 +256, 128)
        self.dec1 = ResidualBlock(128 + 128, 64)
        self.dec0 = ResidualBlock(64+ 64, 64)

        # Bottleneck
        self.bottleneck = ResidualBlock(512,512)

        # Final output
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec3 = self.dec3(torch.cat([self.upsample(bottleneck), enc4], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc3], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc2], dim=1))
        dec0 = self.dec0(torch.cat([self.upsample(dec1), enc1], dim=1))

        # Final output
        out = self.final_conv(dec0)
        return out


class Denoiser(nn.Module):
    def __init__(self,noisy_input_channels, output_channels):
        super(Denoiser, self).__init__()
        self.unet_res =ResidualUNet(noisy_input_channels,output_channels)
    def forward(self,x):
        x = self.unet_res(x)
        return x

