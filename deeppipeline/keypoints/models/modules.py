import torch
import torch.nn.functional as F
from torch import nn

from deeppipeline.common.modules import Identity, conv_block_3x3, conv_block_1x1


class SEBlock(nn.Module):
    def __init__(self, n_features, r=16):
        super(SEBlock, self).__init__()
        self.scorer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(n_features, n_features // r, kernel_size=1, padding=0, stride=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(n_features // r, n_features, kernel_size=1, padding=0, stride=1))

    def forward(self, x):
        return torch.sigmoid(self.scorer(x))*x


class HGResidual(nn.Module):
    def __init__(self, n_inp, n_out, se=False, se_ratio=16):
        super().__init__()
        self.bottleneck = conv_block_1x1(n_inp, n_out // 2, 'relu')
        self.conv = conv_block_3x3(n_out // 2, n_out // 2, 'relu')
        self.out = conv_block_1x1(n_out // 2, n_out, None)

        if n_inp != n_out:
            self.skip = conv_block_1x1(n_inp, n_out, None)
        else:
            self.skip = Identity()

        if se:
            self.se_block = SEBlock(n_out, r=se_ratio)

    def forward(self, x):
        o1 = self.bottleneck(x)
        o2 = self.conv(o1)
        o3 = self.out(o2)
        if hasattr(self, 'se_block'):
            o3 = self.se_block(o3)

        return o3 + self.skip(x)


class MultiScaleHGResidual(nn.Module):
    """
    https://arxiv.org/pdf/1808.04803.pdf

    """

    def __init__(self, n_inp, n_out, se=False, se_ratio=16):
        super().__init__()
        self.scale1 = conv_block_3x3(n_inp, n_out // 2, 'relu')
        self.scale2 = conv_block_3x3(n_out // 2, n_out // 4, 'relu')
        self.scale3 = conv_block_3x3(n_out // 4, n_out - n_out // 4 - n_out // 2, None)

        if n_inp != n_out:
            self.skip = conv_block_1x1(n_inp, n_out, None)
        else:
            self.skip = Identity()
        if se:
            self.se_block = SEBlock(n_out, r=se_ratio)

    def forward(self, x):
        o1 = self.scale1(x)
        o2 = self.scale2(o1)
        o3 = self.scale3(o2)
        o4 = torch.cat([o1, o2, o3], 1)
        if hasattr(self, 'se_block'):
            o4 = self.se_block(o4)

        return o4 + self.skip(x)


class SoftArgmax2D(nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmax2D, self).__init__()
        self.beta = beta

    def forward(self, hm):
        hm = hm.mul(self.beta)
        bs, nc, h, w = hm.size()
        hm = hm.squeeze()

        softmax = F.softmax(hm.view(bs, nc, h * w), dim=2).view(bs, nc, h, w)

        weights = torch.ones(bs, nc, h, w).float().to(hm.device)
        w_x = torch.arange(w).float().div(w)
        w_x = w_x.to(hm.device).mul(weights)

        w_y = torch.arange(h).float().div(h)
        w_y = w_y.to(hm.device).mul(weights.transpose(2, 3)).transpose(2, 3)

        approx_x = softmax.mul(w_x).view(bs, nc, h * w).sum(2).unsqueeze(2)
        approx_y = softmax.mul(w_y).view(bs, nc, h * w).sum(2).unsqueeze(2)

        res_xy = torch.cat([approx_x, approx_y], 2)
        return res_xy


class Hourglass(nn.Module):
    def __init__(self, n, hg_width, n_inp, n_out, upmode='nearest', multiscale_block=False, se=False, se_ratio=16):
        super(Hourglass, self).__init__()

        self.multiscale_block = multiscale_block
        self.upmode = upmode
        self.se = se
        self.se_ratio = se_ratio

        self.lower1 = self.__make_block(n_inp, hg_width)
        self.lower2 = self.__make_block(hg_width, hg_width)
        self.lower3 = self.__make_block(hg_width, hg_width)

        if n > 1:
            self.lower4 = Hourglass(n - 1, hg_width, hg_width, n_out, upmode, se=False, se_ratio=16)
        else:
            self.lower4 = self.__make_block(hg_width, n_out)

        self.lower5 = self.__make_block(n_out, n_out)

        self.upper1 = self.__make_block(n_inp, hg_width)
        self.upper2 = self.__make_block(hg_width, hg_width)
        self.upper3 = self.__make_block(hg_width, n_out)

    def __make_block(self, inp, out):
        if self.multiscale_block:
            return MultiScaleHGResidual(inp, out, self.se, self.se_ratio)
        else:
            return HGResidual(inp, out, self.se, self.se_ratio)

    def forward(self, x):
        o_pooled = F.max_pool2d(x, 2)

        o1 = self.lower1(o_pooled)
        o2 = self.lower2(o1)
        o3 = self.lower3(o2)

        o4 = self.lower4(o3)

        o5 = self.lower5(o4)

        o1_u = self.upper1(x)
        o2_u = self.upper2(o1_u)
        o3_u = self.upper3(o2_u)

        return o3_u + F.interpolate(o5, x.size()[-2:], mode=self.upmode, align_corners=True)
