import torch.nn as nn
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, z):
        xa1 = x + y
        xl1 = self.local_att(xa1)
        xg1 = self.global_att(xa1)
        xlg1 = xl1 + xg1

        xa2 = z + x
        xl2 = self.local_att(xa2)
        xg2 = self.global_att(xa2)
        xlg2 = xl2 + xg2

        xa3 = y + z
        xl3 = self.local_att(xa3)
        xg3 = self.global_att(xa3)
        xlg3 = xl3 + xg3


        wei1 = self.sigmoid(xlg1)
        wei2 = self.sigmoid(xlg2)
        wei3 = self.sigmoid(xlg3)

        xo = x * (wei1 + 1 -wei2) + y * (1 - wei1 + wei3)+ z * (1 - wei3 + wei2)
        return xo

def aff_module(channel,r):
    return AFF(channel,r)