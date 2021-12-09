import torch
import torch.nn.functional as F
from torch import nn


class SimpleTranslator(nn.Module):
    def __init__(self, n, n_styles=18):
        '''
        :param n: Size of noise vector
        :param n_styles: Num of style codes translated from noise (default 18)
        '''
        super(SimpleTranslator, self).__init__()
        self.n = n
        self.n_styles = n_styles

        # Noise (b,n) -> style codes (n_style, 512)
        self.trans = nn.Sequential(
            nn.Linear(n, n * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 2, n * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 2, n * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 4, n * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 4, n * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 8, n * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(n * 8, 512 * n_styles),
            nn.LeakyReLU(0.2),
            nn.Linear(512 * n_styles, 512 * n_styles),
        )

    def forward(self, noise):
        '''
        :param noise: Input noise with shape (b, n)
        :return: Translated style codes with shape (b, n_style, 512)
        '''
        style = self.trans(noise)
        return style.view((-1, self.n_styles, 512))
