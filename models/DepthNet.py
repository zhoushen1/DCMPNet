import torch
import torch.nn as nn
import torch.nn.functional as F

class DRDB(nn.Module):
    def __init__(self, in_ch, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        # print(in_ch_,in_ch)
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out


class DN(nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.DRDB_layer1 = DRDB(in_ch=3, growth_rate=32)
        self.conv1 = nn.Conv2d(3, 24, 3, 2, 1)
        self.DRDB_layer2 = DRDB(in_ch=24, growth_rate=32)
        self.conv2 = nn.Conv2d(24, 48, 3, 2, 1)
        self.DRDB_layer3 = DRDB(in_ch=48, growth_rate=32)
        self.conv3 = nn.Conv2d(48, 96, 3, 2, 1)
        self.DRDB_layer4 = DRDB(in_ch=96, growth_rate=32)
        self.conv4 = nn.Conv2d(96, 128, 3, 2, 1)

        self.DRDB_layer5 = DRDB(in_ch=128, growth_rate=32)
        self.up1 = nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 96, 32, 32]
        self.DRDB_layer6 = DRDB(in_ch=96, growth_rate=32)
        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 48, 64, 64]
        self.DRDB_layer7 = DRDB(in_ch=48, growth_rate=32)
        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 24, 128, 128]
        self.DRDB_layer8 = DRDB(in_ch=24, growth_rate=32)
        self.up4 = nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 3, 256, 256]
        self.final_conv = nn.Conv2d(3,1,1)

    def forward(self, x):

        x1 = self.DRDB_layer1(x)
        x1 = self.conv1(x1)
        x1 = self.DRDB_layer2(x1)
        x1 = self.conv2(x1)
        x1 = self.DRDB_layer3(x1)
        x1 = self.conv3(x1)
        x1 = self.DRDB_layer4(x1)
        x1 = self.conv4(x1)
        x1 = self.DRDB_layer5(x1)
        x1 = self.up1(x1)
        x1 = self.DRDB_layer6(x1)
        x1 = self.up2(x1)
        x1 = self.DRDB_layer7(x1)
        x1 = self.up3(x1)
        x1 = self.DRDB_layer8(x1)
        x1 = self.up4(x1)
        x1 = self.final_conv(x1)

        return x1