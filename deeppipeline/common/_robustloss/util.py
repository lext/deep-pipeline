# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def check_loss_params(alpha_init, scale_init, alpha_lo, alpha_hi, scale_lo):
    if not np.isscalar(alpha_lo):
        raise ValueError('`alpha_lo` must be a scalar, but is of type {}'.format(
            type(alpha_lo)))
    if not np.isscalar(alpha_hi):
        raise ValueError('`alpha_hi` must be a scalar, but is of type {}'.format(
            type(alpha_hi)))
    if alpha_init is not None and not np.isscalar(alpha_init):
        raise ValueError(
            '`alpha_init` must be None or a scalar, but is of type {}'.format(
                type(alpha_init)))
    if not alpha_lo >= 0:
        raise ValueError('`alpha_lo` must be >= 0, but is {}'.format(alpha_lo))
    if not alpha_hi >= alpha_lo:
        raise ValueError('`alpha_hi` = {} must be >= `alpha_lo` = {}'.format(
            alpha_hi, alpha_lo))
    if alpha_init is not None and alpha_lo != alpha_hi:
        if not (alpha_init > alpha_lo and alpha_init < alpha_hi):
            raise ValueError(
                '`alpha_init` = {} must be in (`alpha_lo`, `alpha_hi`) = ({} {})'
                    .format(alpha_init, alpha_lo, alpha_hi))
    if not np.isscalar(scale_lo):
        raise ValueError('`scale_lo` must be a scalar, but is of type {}'.format(
            type(scale_lo)))
    if not np.isscalar(scale_init):
        raise ValueError(
            '`scale_init` must be a scalar, but is of type {}'.format(
                type(scale_init)))
    if not scale_lo > 0:
        raise ValueError('`scale_lo` must be > 0, but is {}'.format(scale_lo))
    if not scale_init >= scale_lo:
        raise ValueError('`scale_init` = {} must be >= `scale_lo` = {}'.format(
            scale_init, scale_lo))


def log_safe(x):
    """The same as torch.log(x), but clamps the input to prevent NaNs."""
    x = torch.as_tensor(x)
    return torch.log(torch.min(x, torch.tensor(33e37).to(x)))


def log1p_safe(x):
    """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
    x = torch.as_tensor(x)
    return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))


def exp_safe(x):
    """The same as torch.exp(x), but clamps the input to prevent NaNs."""
    x = torch.as_tensor(x)
    return torch.exp(torch.min(x, torch.tensor(87.5).to(x)))


def expm1_safe(x):
    """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
    x = torch.as_tensor(x)
    return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))


def inv_softplus(y):
    """The inverse of tf.nn.softplus()."""
    y = torch.as_tensor(y)
    return torch.where(y > 87.5, y, torch.log(torch.expm1(y)))


def logit(y):
    """The inverse of tf.nn.sigmoid()."""
    y = torch.as_tensor(y)
    return -torch.log(1. / y - 1.)


def affine_sigmoid(logits, lo=0, hi=1):
    """Maps reals to (lo, hi), where 0 maps to (lo+hi)/2."""
    if not lo < hi:
        raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
    logits = torch.as_tensor(logits)
    lo = torch.as_tensor(lo)
    hi = torch.as_tensor(hi)
    alpha = torch.sigmoid(logits) * (hi - lo) + lo
    return alpha


def inv_affine_sigmoid(probs, lo=0, hi=1):
    """The inverse of affine_sigmoid(., lo, hi)."""
    if not lo < hi:
        raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
    probs = torch.as_tensor(probs)
    lo = torch.as_tensor(lo)
    hi = torch.as_tensor(hi)
    logits = logit((probs - lo) / (hi - lo))
    return logits


def affine_softplus(x, lo=0, ref=1):
    """Maps real numbers to (lo, infinity), where 0 maps to ref."""
    if not lo < ref:
        raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
    x = torch.as_tensor(x)
    lo = torch.as_tensor(lo)
    ref = torch.as_tensor(ref)
    shift = inv_softplus(torch.tensor(1.))
    y = (ref - lo) * torch.nn.Softplus()(x + shift) + lo
    return y


def inv_affine_softplus(y, lo=0, ref=1):
    """The inverse of affine_softplus(., lo, ref)."""
    if not lo < ref:
        raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
    y = torch.as_tensor(y)
    lo = torch.as_tensor(lo)
    ref = torch.as_tensor(ref)
    shift = inv_softplus(torch.tensor(1.))
    x = inv_softplus((y - lo) / (ref - lo)) - shift
    return x


def compute_jacobian(f, x):
    """Computes the Jacobian of function `f` with respect to input `x`."""
    vec = lambda z: torch.reshape(z, [-1])
    jacobian = []
    for i in range(np.prod(x.shape)):
        var_x = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        y = vec(f(var_x))[i]
        y.backward()
        jacobian.append(np.array(vec(var_x.grad)))
    jacobian = np.stack(jacobian, 1)
    return jacobian


def get_resource_as_file(path):
    """A uniform interface for internal/open-source files."""

    class NullContextManager(object):

        def __init__(self, dummy_resource=None):
            self.dummy_resource = dummy_resource

        def __enter__(self):
            return self.dummy_resource

        def __exit__(self, *args):
            pass

    return NullContextManager(path)


def get_resource_filename(path):
    """A uniform interface for internal/open-source filenames."""
    return path
