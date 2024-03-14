import math
import torch.nn.functional as F
from torch import nn
import torch

from torch import nn


class Atten(nn.Module):
    def __init__(self, channel):
        super(Atten, self).__init__()

        self.activation = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
        )

        self.att_conv1 = nn.Sequential(
                nn.Conv2d(channel * 3, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
        )
        self.att_conv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
        )
        self.att_conv3 = nn.Sequential(
                nn.Conv2d(channel, 3, kernel_size=1),
                nn.Sigmoid(),
        )

        # nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)

    def forward(self, x, y, z):
        n, _, h, w = x.shape
        # fusion
        tmp = torch.cat([x, y, z], dim=1)
        att_w = self.att_conv1(tmp)
        att_w = self.activation(att_w)
        att_w = self.att_conv2(att_w)
        att_w = self.activation(att_w)
        att_w = self.att_conv3(att_w)
        out = (x * att_w[:, 0].view(n, 1, h, w) + y * att_w[:, 1].view(n, 1, h, w) +
               z * att_w[:, 2].view(n, 1, h, w))
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.activation(out)
        return out



class Atten1(nn.Module):
    def __init__(self, channel):
        super(Atten1, self).__init__()
        self.att_conv = nn.Sequential(
                nn.Conv2d(channel * 3, 3, kernel_size=1),
                nn.Sigmoid(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.activation = nn.ReLU()
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(channel * 2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(channel),
        # )

    def forward(self, x, y, z):
        n, _, h, w = x.shape
        # fusion
        tmp = torch.cat([x, y, z], dim=1)
        att_w = self.att_conv(tmp)
        att_w = self.activation(att_w)
        out = (x * att_w[:, 0].view(n, 1, h, w) + y * att_w[:, 1].view(n, 1, h, w) +
               z * att_w[:, 2].view(n, 1, h, w))
        out = self.conv1(out)
        out = self.activation(out)
        return out

def atten_module(channel):
    return Atten(channel)

def atten_module1(channel):
    return Atten1(channel)
