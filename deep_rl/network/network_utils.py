#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *

class BaseNet:
    def __init__(self):
        pass
        
    def loss(self): # for loss collection
        tot_loss = self._loss if hasattr(self, '_loss') else 0.0
        for child in self.children():
            if hasattr(child, 'loss'):
                tot_loss += child.loss()
        return tot_loss

class MultiLinear(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, key):
        super().__init__()
        self.weights = nn.Parameter(weight_init(torch.randn(n_heads, input_dim, output_dim)))
        self.biases = nn.Parameter(torch.zeros(n_heads, output_dim))
        self.key = key

    def forward(self, inputs, info):
        weights = self.weights[tensor(info[self.key], torch.int64),:,:]
        biases = self.biases[tensor(info[self.key], torch.int64),:]
        output = torch.bmm(inputs.unsqueeze(1), weights).squeeze(1) + biases
        return output

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def weight_init(weight, w_scale=1.0):
    nn.init.orthogonal_(weight.data)
    weight.data.mul_(w_scale)
    return weight

#######################################################################
# Common functions for buidling network architecture
#######################################################################

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(layer_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = layer_init(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out
