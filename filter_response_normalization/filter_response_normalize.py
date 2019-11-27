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

class FilterResponseNorm1d(nn.Module):
    _version = 2
    __constants__ = ['tau', 'beta', 'gamma', 'eps']

    def __init__(self, num_features, eps=1e-5, eps_learnable=True):
        super( FilterResponseNorm1d, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        self.gamma = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        if eps_learnable: # if eps is learnable parameter
            self.eps = nn.parameter.Parameter(torch.Tensor(1))
            nn.init.constant_(self.eps, eps)
        else:
            self.eps = eps
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
#         super(FilterResponseNorm2d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        nu2 = torch.mean(input**2, axis=[2], keepdims=True)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps))
        return self.gamma * input + self.beta

class FilterResponseNorm2d(nn.Module):
    _version = 2
    __constants__ = ['tau', 'beta', 'gamma', 'eps']

    def __init__(self, num_features, eps=1e-5, eps_learnable=True):
        super(FilterResponseNorm2d, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1))
        if eps_learnable: # if eps is learnable parameter
            self.eps = nn.parameter.Parameter(torch.Tensor(1))
            nn.init.constant_(self.eps, eps)
        else:
            self.eps = eps
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
#         super(FilterResponseNorm2d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        nu2 = torch.mean(input**2, axis=[2, 3], keepdims=True)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps))
        return self.gamma * input + self.beta
    
class ThresholdedLinearUnit(nn.Module):
    _version = 2
    __constants__ = ['tau']
    
    def __init__(self, inplace=True):
        super(ThresholdedLinearUnit, self).__init__()
        self.inplace = inplace
        self.tau = nn.parameter.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        
    def forward(self, input):
        if self.inplace:
            input = torch.max(input, self.tau)
            return input
        else:
            return torch.max(input, self.tau)

class ThresholdedLinearUnitFix1D(nn.Module):
    _version = 2
    __constants__ = ['tau']
    
    def __init__(self, num_features, inplace=True):
        super(ThresholdedLinearUnitFix, self).__init__()
        self.inplace = inplace
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        
    def forward(self, input):
        if self.inplace:
            input = torch.max(input, self.tau)
            return input
        else:
            return torch.max(input, self.tau)
        
class ThresholdedLinearUnitFix2D(nn.Module):
    _version = 2
    __constants__ = ['tau']
    
    def __init__(self, num_features, inplace=True):
        super(ThresholdedLinearUnitFix, self).__init__()
        self.inplace = inplace
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        
    def forward(self, input):
        if self.inplace:
            input = torch.max(input, self.tau)
            return input
        else:
            return torch.max(input, self.tau)
    
        
def convert_model(module):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d
       to SynchronizedBatchNorm*N*d

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
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = DataParallelWithCallback(mod)
        return mod

    mod = module
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        mod = FilterResponseNorm1d(module.num_features, module.eps)
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        mod = FilterResponseNorm2d(module.num_features, module.eps)
    elif isinstance(module, torch.nn.ReLU):
        mod = ThresholdedLinearUnit(module.inplace)
    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))
    return mod

