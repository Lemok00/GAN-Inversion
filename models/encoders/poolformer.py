import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torchvision.models.resnet import resnet152

import sys

sys.path.append('.')
sys.path.append('..')
from .map2style import Map2StyleBlock


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_channel=3, embed_dim=768, norm_layer=None):
        super().__init__()

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):

    def __init__(self, embed_dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=ChannelLayerNorm, drop=0.0):
        super().__init__()

        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def define_block_groups(embed_dim, num_layers, pool_size=3, mlp_ratio=4.,
                        act_layer=nn.GELU, norm_layer=ChannelLayerNorm, drop_rate=.0):
    blocks = []
    for block_idx in range(num_layers):
        blocks.append(PoolFormerBlock(
            embed_dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate)
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer_StyleEncoder(nn.Module):
    def __init__(self, in_channel, size=256, n_styles=18):
        super().__init__()

        self.num_layers = [8, 8, 24, 8]
        self.embed_dims = [96, 192, 384, 768]
        self.mlp_ratios = [4, 4, 4, 4]
        self.downsamples = [True, True, True, True]

        self.input_size = size
        self.style_count = n_styles

        self.network = nn.ModuleList()
        for i in range(len(self.num_layers)):
            network_stage = []
            if i == 0:
                network_stage.append(PatchEmbedding(patch_size=7, stride=4, padding=2,
                                                    in_channel=in_channel, embed_dim=self.embed_dims[0]))
            elif not i == 0 and (self.downsamples[i - 1] or not self.embed_dims[i - 1] == self.embed_dims[i]):
                network_stage.append(PatchEmbedding(patch_size=3, stride=2, padding=1,
                                                    in_channel=self.embed_dims[i - 1], embed_dim=self.embed_dims[i]))
            block_group = define_block_groups(self.embed_dims[i], self.num_layers[i], mlp_ratio=self.mlp_ratios[i])
            network_stage.append(block_group)

            self.network.append(nn.Sequential(*network_stage))

        self.styles = nn.ModuleList()
        self.coarse_start = 3
        self.mid_start = 6
        self.fine_start = 11

        for i in range(n_styles):
            if i < self.coarse_start:
                style = Map2StyleBlock(self.embed_dims[3], 512, self.input_size / 32)
            elif i < self.mid_start:
                style = Map2StyleBlock(self.embed_dims[2], 512, self.input_size / 16)
            elif i < self.fine_start:
                style = Map2StyleBlock(self.embed_dims[1], 512, self.input_size / 8)
            else:
                style = Map2StyleBlock(self.embed_dims[0], 512, self.input_size / 4)
            self.styles.append(style)

    def upsample_and_add(self, x, y):
        b, c, h, w = y.shape
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        feats = []
        for idx, stage in enumerate(self.network):
            x = stage(x)
            feats.append(x)

        latents = []

        for idx in range(self.coarse_start):
            latents.append(self.styles[idx](feats[3]))

        for idx in range(self.coarse_start, self.mid_start):
            latents.append(self.styles[idx](feats[2]))

        for idx in range(self.mid_start, self.fine_start):
            latents.append(self.styles[idx](feats[1]))

        for idx in range(self.fine_start, self.style_count):
            latents.append(self.styles[idx](feats[0]))

        # [(b, 512),...,(b, 512)] -> (b, 14, 512)
        out = torch.stack(latents, dim=1)
        return out
