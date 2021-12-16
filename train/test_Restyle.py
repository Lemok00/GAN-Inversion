import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import utils, transforms

from argparse import ArgumentParser
import math
import os

import dnnlib
import legacy

import time
from mytime import time_change
from dataset import set_dataset

import sys

sys.path.append('.')
sys.path.append('..')
from configs.model_config import encoder_list, pretrained_information

from criteria.lpips.lpips import LPIPS

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name of pretrained StyleGAN2 (e.g. FFHQ256).')
    parser.add_argument("--dataset_path", type=str, required=True,
                        help='Path of the dataset.')
    parser.add_argument("--dataset_type", choices=['resized_lmdb'],
                        help='Type of the dataset (e.g. resized_lmdb).')

    parser.add_argument('--n_steps', type=int, default=3,
                        help='Steps in a iteration.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data.')
    parser.add_argument('--ckpt', type=int, default=4,
                        help='Checkpoint to be loaded.')

    parser.add_argument('--encoder', type=str, choices=['resnet34', 'resnet152', 'poolformer'], default='resnet152',
                        help='The encoder for encoding images into W+ latent codes (resnet152).')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = args.device

    base_dir = f"../experiments/{args.encoder}_{args.name}_steps{args.n_steps}"
    ckpt_dir = f'{base_dir}/ckpt'
    result_dir = f'{base_dir}/result'

    ckpt = torch.load(f'c{ckpt_dir}/{args.ckpt}.pth')

    with open(f"{base_dir}/testing_config.txt", "wt") as fp:
        for k, v in vars(args).items():
            fp.write(f'{k}: {v}\n')
        fp.close()

    image_size = pretrained_information[args.name]['size']
    pretrained_path = pretrained_information[args.name]['path']

    # The number of StyleBlocks to generate image with the size
    # 14 StyleBlocks for 256x256 images
    n_styles = int(math.log(image_size, 2)) * 2 - 2

    # Load weights of pretrained StyleGAN2 Generator
    print(f'Loading weights from pretrained StyleGAN2 ({args.name}) ...')
    with dnnlib.util.open_url(pretrained_path) as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    generator.train()
    print(f'pretrained StyleGAN2 ({args.name}) on {device} loaded.\n')

    # Load Encoder
    print(f'Loading Encoder ...')
    encoder = encoder_list[args.encoder](in_channel=6, size=image_size, n_styles=n_styles).to(device)
    encoder.load_state_dict(ckpt)
    encoder.train()
    print(f'Pretrained Encoder loaded.\n')

    # Initialize Datasets
    print('Loading datasets ...')
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    # train_dataset = set_dataset(args.dataset_type, args.dataset_path, transform,
    #                             image_size, False, args.train_ratio)
    test_dataset = set_dataset(args.dataset_type, args.dataset_path, transform,
                               image_size, True, args.train_ratio)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Datasets Loaded.\n')

    calculate_lpips_loss = LPIPS(net_type='vgg').to(device)

    # Average w
    w_avg = generator.mapping.w_avg
    # print(w_avg.shape) #(512)
    average_style = w_avg.repeat(n_styles, 1).unsqueeze(0)
    average_image = generator.synthesis(average_style, noise_mode='none').repeat(args.batch_size, 1, 1, 1).detach()

    # utils.save_image(average_image, f'{base_dir}/average_image.png', range=(-1, 1), normalize=True)

    # Testing
    print('Start Testing\n')

    start_time = time.time()

    total_nums = 0
    mse_loss_sum = 0
    lpips_loss_sum = 0

    for iter_idx, real_image in enumerate(test_loader):
        real_image = real_image.to(device)

        this_batch_size = real_image.shape[0]
        if not this_batch_size == args.batch_size:
            continue
        recovered_image = average_image
        recovered_style = average_style.repeat(args.batch_size, 1, 1)

        total_nums += this_batch_size

        recovered_images = []

        with torch.no_grad():
            for restyle_idx in range(args.n_steps):
                x_input = torch.cat([real_image, recovered_image.clone().detach()], dim=1)
                recovered_style = encoder(x_input) + recovered_style.clone().detach()
                recovered_image = generator.synthesis(recovered_style)

        # Reconstruction Loss
        recon_loss = F.mse_loss(recovered_image, real_image)
        # LPIPS Loss
        lpips_loss = calculate_lpips_loss(recovered_image, real_image)

        mse_loss_sum+=recon_loss.item()
        lpips_loss_sum+=lpips_loss.item()

        recovered_images.append(recovered_image)

        saving_image = torch.cat(recovered_images, dim=-1)
        for i in range(saving_image.shape[0]):
            utils.save_image(saving_image[i],
                             f'{result_dir}/{iter_idx*this_batch_size+i:06d}.png',
                             normalize=True,
                             range=(-1, 1))
