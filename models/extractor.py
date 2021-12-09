import torch
import torch.nn.functional as F
from torch import nn


class SimpleExtractor(nn.Module):
    def __init__(self, n, n_styles=18):
        '''
        :param n: Size of noise vector extracted from style codes
        :param n_styles: Num of style codes (default 18)
        '''
        super(SimpleExtractor, self).__init__()
        self.n = n
        self.n_styles = n_styles

        # Style codes (b, n_style * 512) -> Noise (b, n)
        self.extract = nn.Sequential(
            nn.Linear(512 * n_styles, n * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 8, n * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 4, n * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 2, n),
            nn.Tanh()
        )

    def forward(self, style):
        '''
        :param style: Input style codes with shape (b, n_style, 512)
        :return: Extracted noises with shape (b, n)
        '''
        style = style.view(-1, self.n_styles * 512)
        noise = self.extract(style)
        return noise
