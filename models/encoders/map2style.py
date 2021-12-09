import numpy as np
from torch import nn
import torch
import sys

sys.path.append('.')
sys.path.append('..')
#from ..stylegan2.model import EqualLinear


class Map2StyleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resolution):
        '''
        :param in_channel: channel of input feature map
        :param out_channel: size of output vector
        :param resolution: resolution of input feature map
        '''
        super(Map2StyleBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.resolution = resolution
        # Num of /2 downsample conv pools
        self.num_pools = int(np.log2(resolution))

        # Layers in Map2StyleBlock
        modules = []
        modules += [nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(self.num_pools - 1):
            modules += [nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU()]
        self.convs = nn.Sequential(*modules)
        #self.linear = EqualLinear(out_channel, out_channel, lr_mul=1)
        self.linear = nn.Linear(out_channel, out_channel)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: output latent code
        '''
        x = self.convs(x)
        x = x.view(-1, self.out_channel)
        x = self.linear(x)

        return x
