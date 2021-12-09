import torch
import torch.nn.functional as F
from torch import nn


class SimpleDiscriminator(nn.Module):
    def __init__(self, n_styles=18):
        '''
        :param n_styles: Num of style codes (default 18)
        '''
        super(SimpleDiscriminator, self).__init__()
        self.n_styles = n_styles

        # Style codes (b, n_style, 512) -> label (b,1)
        self.discriminate = nn.Sequential(
            nn.Linear(512 * n_styles, 64 * n_styles),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * n_styles, 8 * n_styles),
            nn.LeakyReLU(0.2),
            nn.Linear(8 * n_styles, 1 * n_styles),
            nn.LeakyReLU(0.2),
            nn.Linear(1 * n_styles, 1)
        )

    def forward(self, style_codes):
        '''
        :param style: Input style codes with shape (b, n_style, 512)
        :return: Discriminated label with shape (b,)
        '''

        noise = self.discriminate(style_codes.view((-1, self.n_styles * 512))).view(-1, )
        return noise
