from deeppipeline.common.modules import Identity, conv_block_3x3, conv_block_1x1
import torch
from torch import nn
import torch.nn.functional as F


class HGResidual(nn.Module):
    def __init__(self, n_inp, n_out):
        super().__init__()
        self.bottleneck = conv_block_1x1(n_inp, n_out // 2, 'relu')
        self.conv = conv_block_3x3(n_out // 2, n_out // 2, 'relu')
        self.out = conv_block_1x1(n_out // 2, n_out, None)

        if n_inp != n_out:
            self.skip = conv_block_1x1(n_inp, n_out, None)
        else:
            self.skip = Identity()

    def forward(self, x):
        o1 = self.bottleneck(x)
        o2 = self.conv(o1)
        o3 = self.out(o2)

        return o3 + self.skip(x)


class MultiScaleHGResidual(nn.Module):
    """
    https://arxiv.org/pdf/1808.04803.pdf

    """
    def __init__(self, n_inp, n_out):
        super().__init__()
        self.scale1 = conv_block_3x3(n_inp, n_out // 2, 'relu')
        self.scale2 = conv_block_3x3(n_out // 2, n_out // 4, 'relu')
        self.scale3 = conv_block_3x3(n_out // 4, n_out - n_out // 4 - n_out // 2, None)

        if n_inp != n_out:
            self.skip = conv_block_1x1(n_inp, n_out, None)
        else:
            self.skip = Identity()

    def forward(self, x):
        o1 = self.scale1(x)
        o2 = self.scale2(o1)
        o3 = self.scale3(o2)

        return torch.cat([o1, o2, o3], 1) + self.skip(x)


class SoftArgmax2D(nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmax2D, self).__init__()
        self.beta = beta

    def forward(self, hm):
        hm = hm.mul(self.beta)
        bs, nc, h, w = hm.size()
        hm = hm.squeeze()

        softmax = F.softmax(hm.view(bs, nc, h*w), dim=2).view(bs, nc, h, w)

        weights = torch.ones(bs, nc, h, w).float().to(hm.device)
        w_x = torch.arange(w).float().div(w)
        w_x = w_x.to(hm.device).mul(weights)

        w_y = torch.arange(h).float().div(h)
        w_y = w_y.to(hm.device).mul(weights.transpose(2, 3)).transpose(2, 3)

        approx_x = softmax.mul(w_x).view(bs, nc, h*w).sum(2).unsqueeze(2)
        approx_y = softmax.mul(w_y).view(bs, nc, h*w).sum(2).unsqueeze(2)

        res_xy = torch.cat([approx_x, approx_y], 2)
        return res_xy


class Hourglass(nn.Module):
    def __init__(self, n, hg_width, n_inp, n_out, upmode='nearest', multiscale_block=False):
        super(Hourglass, self).__init__()

        self.multiscale_block = multiscale_block
        self.upmode = upmode

        self.lower1 = self.__make_block(n_inp, hg_width)
        self.lower2 = self.__make_block(hg_width, hg_width)
        self.lower3 = self.__make_block(hg_width, hg_width)

        if n > 1:
            self.lower4 = Hourglass(n - 1, hg_width, hg_width, n_out, upmode)
        else:
            self.lower4 = self.__make_block(hg_width, n_out)

        self.lower5 = self.__make_block(n_out, n_out)

        self.upper1 = self.__make_block(n_inp, hg_width)
        self.upper2 = self.__make_block(hg_width, hg_width)
        self.upper3 = self.__make_block(hg_width, n_out)

    def __make_block(self, inp, out):
        if self.multiscale_block:
            return MultiScaleHGResidual(inp, out)
        else:
            return HGResidual(inp, out)

    def forward(self, x):
        o_pooled = F.max_pool2d(x, 2)

        o1 = self.lower1(o_pooled)
        o2 = self.lower2(o1)
        o3 = self.lower3(o2)

        o4 = self.lower4(o3)

        o1_u = self.upper1(x)
        o2_u = self.upper2(o1_u)
        o3_u = self.upper3(o2_u)
        return o3_u + F.interpolate(self.lower5(o4), x.size()[-2:], mode=self.upmode, align_corners=True)

