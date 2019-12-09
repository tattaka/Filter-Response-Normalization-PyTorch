# -*- coding: utf-8 -*-
# File   : filter_response_normalization.py
# Author : tattaka
# Email  : tattaka666@gmail.com
# Date   : 27/11/2019
#
# This file is part of Filter-Response-Normalization-PyTorch
# https://github.com/tattaka/Filter-Response-Normalization-PyTorch
# Distributed under MIT License.

import torch
from torch import nn

from .replicate import DataParallelWithCallback

__all__ = [
    'FilterResponseNorm1d', 'FilterResponseNorm2d', 'convert_model'
]

## Big fix for https://github.com/tattaka/Filter-Response-Normalization-PyTorch/issues/2
class _FilterResponseNorm(nn.Module):
    __constants__ = ["num_features", "eps", "eps_trainable", "tau", "beta", "gamma"]

    def __init__(self, shape, activated=True, eps=1e-6, eps_trainable=True):
        super(_FilterResponseNorm, self).__init__()
        self._eps = eps
        self.activated = activated
        self.num_features = shape[1]
        self.eps_trainable = eps_trainable

        self.beta = nn.Parameter(torch.zeros(shape))
        self.gamma = nn.Parameter(torch.ones(shape))

        if self.eps_trainable:
            self.eps = nn.Parameter(torch.full(shape, eps))
        else:
            self.eps = eps

        if self.activated:
            self.tau = nn.Parameter(torch.zeros(shape))
        else:
            self.tau = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        if isinstance(self.eps, nn.Parameter):
            nn.init.constant_(self.eps, self._eps)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        nu2 = torch.mean(input ** 2, axis=2, keepdims=True)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps) + self._eps)
        output = self.gamma * input + self.beta
        if self.activated:
            output = torch.max(output, self.tau)
        return output


class FilterResponseNorm1d(_FilterResponseNorm):

    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm1d, self).__init__(
            shape=(1, num_features, 1),
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class FilterResponseNorm2d(_FilterResponseNorm):

    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm2d, self).__init__(
            shape=(1, num_features, 1, 1),
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

def convert_model(module):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d + ReLU()
       to FilterResponseNorm*N*d
    Args:
        module: the input module needs to be convert to FRN model
    Examples:
        >>> import torch.nn as nn
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = nn.DataParallel(m)
        >>> # after convert, m is using FRN
        >>> m = convert_model(m)
    """

    mod = module
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        mod = FilterResponseNorm1d(module.num_features, activated=True, eps=module.eps)
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        mod = FilterResponseNorm2d(module.num_features, activated=True, eps=module.eps)
    elif isinstance(module, torch.nn.ReLU):
        mod = torch.nn.Identity()
    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))
    return mod