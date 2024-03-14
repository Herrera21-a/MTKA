
import torch.nn as nn


import torch
import torch.nn.functional as F

def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MSCA(nn.Module):
    def __init__(self, channels, r):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei


class ACFM(nn.Module):
    def __init__(self,channels, r):
        super(ACFM, self).__init__()

        self.msca = MSCA(channels,r)
        self.upsample = cus_sample
        self.conv = BasicConv2d(in_planes=channels, out_planes=channels, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y, z):
        y1 = self.upsample(y, scale_factor=1)
        # print(x.shape)
        # print(y.shape)
        # print(y1.shape)

        xy = x + y1

        wei1 = self.msca(xy)

        x1 = self.upsample(x, scale_factor=1)
        zx = z + x1
        wei2 = self.msca(zx)

        z1 = self.upsample(z, scale_factor=1)
        yz = y + z1
        wei3 = self.msca(yz)

        xo = x * (1-wei2+ wei1 ) + y * (1 - wei1+wei3)+ z * (1 - wei3+wei2)
        xo = self.conv(xo)

        return xo

def acfm_module(channel,r):
    return ACFM(channel,r)





