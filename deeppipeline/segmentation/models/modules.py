import torch
import torch.nn as nn
import torch.nn.functional as F

from deeppipeline.common.modules import conv_block_3x3


class Encoder(nn.Module):
    """Encoder block. For encoder-decoder architecture.
    Conv3x3-Conv3x3-Maxpool 2x2. BatchNorm or other form of normalization is used optionally.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, inp_channels, out_channels, depth=2, activation='relu', normalization='BN'):
        super().__init__()
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(conv_block_3x3(inp_channels, out_channels, activation, normalization))
            else:
                tmp.append(conv_block_3x3(out_channels, out_channels, activation, normalization))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder block. For encoder-decoder architecture.
    Bilinear ups->Conv3x3-Conv3x3. BatchNorm or other form of normalization is used optionally.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, inp_channels, out_channels, depth=2, mode='bilinear', activation='relu', normalization='BN'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(conv_block_3x3(inp_channels, out_channels, activation, normalization))
            else:
                tmp.append(conv_block_3x3(out_channels, out_channels, activation, normalization))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.size()[-2:], mode=self.ups_mode, align_corners=True)
        y = torch.cat([x_ups, x_big], 1)
        y = self.layers(y)
        return y
