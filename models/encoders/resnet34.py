import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torchvision.models.resnet import resnet34

import sys

sys.path.append('.')
sys.path.append('..')
from .map2style import Map2StyleBlock

class ResNet34_StyleEncoder(nn.Module):
    def __init__(self, in_channel, size=256, n_styles=18):
        '''
        :param in_channel: Channel of input image (6 for RGBRGB)
        :param n_styles: Num of style layers in generator (18 for StyleGAN 2)
        :param size: H(W) of input image (default 256)
        '''
        super(ResNet34_StyleEncoder, self).__init__()
        self.in_channel = in_channel
        self.input_size = size
        self.style_count = n_styles

        # Input shape: (c, h, w)
        # Conv_1 in ResNet: 7x7, 64, /2
        self.conv_1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.PReLU(64)
        # Feature map: (64, h/2, w/2)

        # Pretrained ResNet on ImageNet
        resnet_basenet = resnet34(pretrained=False)
        # Conv_2~5 in ResNet
        resblocks = [
            resnet_basenet.layer1,  # Output feature map: (64, h/2, w/2)
            resnet_basenet.layer2,  # Output feature map: (128, h/4, w/4)
            resnet_basenet.layer3,  # Output feature map: (256, h/8, w/8)
            resnet_basenet.layer4  # Output feature map: (512, h/16, w/16)
        ]
        # Gradually append layers
        modules = []
        for resblock in resblocks:
            for layer in resblock:
                modules.append(layer)
        # Main body of the Encoder
        self.body = nn.Sequential(*modules)
        # Feature Map: (512, h/16, w/16)

        self.styles = nn.ModuleList()
        # Style blocks control coarse features: 0~2
        self.mid_start = 3
        # Style blocks control middle features: 3~6
        self.fine_start = 7
        # Style blocks control fine features: 7~18

        # Add Map2Style Layers
        for i in range(n_styles):
            if i < self.mid_start:
                style = Map2StyleBlock(512, 512, self.input_size / 16)
            elif i < self.mid_start:
                style = Map2StyleBlock(512, 512, self.input_size / 8)
            else:
                style = Map2StyleBlock(512, 512, self.input_size / 4)
            self.styles.append(style)

        # 1x1 Convolutional Layer for concating feature maps in different layer
        self.trans_conv_1 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.trans_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)

    def upsample_and_add(self, x, y):
        '''
        :param x: Feature map with low resolution (from coarser layer)
        :param y: Feature map with high resolution (from finer layer)
        :return: upsampled x + y
        '''

        b, c, h, w = y.shape
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        '''
        :param x: Target image concated with re-generated image x'
        :return: latent codes for generating image. Shape = (b, 18, 512)
        '''
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        latents = []
        modulelist = list(self.body._modules.values())
        # Process the feature map by each convolution layer
        for idx, layer in enumerate(modulelist):
            x = layer(x)
            if idx == 6:
                # Output for first layer in Conv_3
                # Feature Map: (128, h/4, w/4)
                fm_1 = x
            elif idx == 12:
                # Output for first layer in Conv_4
                # Feature Map: (256, h/8, w/8)
                fm_2 = x
            elif idx == 15:
                # Output for first layer in Conv_5
                # Feature Map: (512, h/16, w/16)
                fm_3 = x

        # Feature map to Style
        # Coarse feature to Style (0~2)
        for idx in range(self.mid_start):
            latents.append(self.styles[idx](fm_3))
        # Upsample(fm_3) + Updimension(fm_2)
        plus_2 = self.upsample_and_add(fm_3, self.trans_conv_2(fm_2))
        # Middle feature to Style (3~6)
        for idx in range(self.mid_start, self.fine_start):
            latents.append(self.styles[idx](plus_2))
        # Upsample(p_2) + Updimension(fm_1)
        plus_1 = self.upsample_and_add(plus_2, self.trans_conv_1(fm_1))
        # Fine feature to Style (7~17)
        for idx in range(self.fine_start, self.style_count):
            latents.append(self.styles[idx](plus_1))

        # [(b, 512),...,(b, 512)] -> (b, 18, 512)
        out = torch.stack(latents, dim=1)
        return out


class SimpleResNet34_StyleEncoder(nn.Module):
    def __init__(self, in_channel, size=256, n_styles=18):
        '''
        :param in_channel: Channel of input image (6 for RGBRGB)
        :param n_styles: Num of style layers in generator (18 for StyleGAN 2)
        :param size: H(W) of input image (default 256)
        '''
        super(SimpleResNet34_StyleEncoder, self).__init__()
        self.in_channel = in_channel
        self.input_size = size
        self.style_count = n_styles

        # Input shape: (c, h, w)
        # Conv_1 in ResNet: 7x7, 64, /2
        self.conv_1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.PReLU(64)
        # Feature map: (64, h/2, w/2)

        # Pretrained ResNet on ImageNet
        resnet_basenet = resnet34(pretrained=True)
        # Conv_2~5 in ResNet
        resblocks = [
            resnet_basenet.layer1,  # Output feature map: (64, h/2, w/2)
            resnet_basenet.layer2,  # Output feature map: (128, h/4, w/4)
            resnet_basenet.layer3,  # Output feature map: (256, h/8, w/8)
            resnet_basenet.layer4  # Output feature map: (512, h/16, w/16)
        ]
        # Gradually append layers
        modules = []
        for resblock in resblocks:
            for layer in resblock:
                modules.append(layer)
        # Main body of the Encoder
        self.body = nn.Sequential(*modules)
        # Feature Map: (512, h/16, w/16)

        self.styles = nn.ModuleList()

        # Add Map2Style Layers
        for i in range(n_styles):
            self.styles.append(Map2StyleBlock(512, 512, size / 16))

    def forward(self, x):
        '''
        :param x: Target image concated with re-generated image x'
        :return: latent codes for generating image. Shape = (b, 18, 512)
        '''
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.body(x)
        latents = []
        for i in range(self.style_count):
            latents.append(self.styles[i](x))

        out = torch.stack(latents, dim=1)
        return out
