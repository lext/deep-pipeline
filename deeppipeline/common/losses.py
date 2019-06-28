import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ._robustloss.adaptiveloss import AdaptiveLossFunction


class ElasticLoss(nn.Module):
    def __init__(self, w=0.5):
        super(ElasticLoss, self).__init__()
        self.weights = torch.FloatTensor([w, 1 - w])

    def forward(self, preds, gt):
        loss = 0

        if not isinstance(preds, tuple):
            preds = (preds,)

        for i in range(len(preds)):
            l2 = F.mse_loss(preds[i].squeeze(), gt.squeeze()).mul(self.weights[0])
            l1 = F.l1_loss(preds[i].squeeze(), gt.squeeze()).mul(self.weights[1])
            loss += l1 + l2

        return loss


class LNLoss(nn.Module):
    def __init__(self, space='l2'):
        super(LNLoss, self).__init__()
        if space == 'l2':
            self.loss = F.mse_loss
        elif space == 'l1':
            self.loss = F.l1_loss
        else:
            raise NotImplementedError

    def forward(self, preds, gt):
        loss = 0

        if not isinstance(preds, tuple):
            preds = (preds,)

        for i in range(len(preds)):
            loss += self.loss(preds[i].squeeze(), gt.squeeze())

        return loss


class WingLoss(nn.Module):
    """
    https://arxiv.org/pdf/1711.06753.pdf

    Refactored implementation from  https://github.com/BloodAxe/pytorch-toolbelt/.
    Supports intermediate supervision.

    """

    def __init__(self, width=5, curvature=0.5):
        super(WingLoss, self).__init__()
        self.w = width
        self.curvature = curvature
        self.c = self.w - self.w * math.log(1 + self.w / self.curvature)

    def __forward_single(self, preds, target):
        diff_abs = (target - preds).abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.w
        idx_bigger = diff_abs >= self.w

        loss[idx_smaller] = self.w * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger] = loss[idx_bigger] - self.c

        return loss.mean()

    def forward(self, preds, target):
        loss = 0

        if not isinstance(preds, tuple):
            preds = (preds,)

        for i in range(len(preds)):
            loss += self.__forward_single(preds[i], target)

        return loss


class GeneralizedRobustLoss(nn.Module):
    def __init__(self,
                 num_dims,
                 alpha_lo: float = 0.001,
                 alpha_hi: float = 1.999,
                 alpha_init=None,
                 scale_lo: float = 1e-5,
                 scale_init: float = 1.0,
                 float_dtype=np.float32,
                 device: str = 'cuda'):

        super(GeneralizedRobustLoss, self).__init__()
        self.loss_obj = AdaptiveLossFunction(num_dims, alpha_lo, alpha_hi, alpha_init,
                                             scale_lo, scale_init, float_dtype, device)

    def forward(self, preds, target):
        loss = 0

        if not isinstance(preds, tuple):
            preds = (preds,)

        for i in range(len(preds)):
            loss += self.loss_obj(preds[i], target)

        return loss
