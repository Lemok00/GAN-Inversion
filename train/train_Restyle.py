import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import utils, transforms
from torchsummary import summary

from argparse import ArgumentParser
import math
import os
import numpy as np

import dnnlib
import legacy

import time
from mytime import time_change
from dataset import set_dataset

import sys

sys.path.append('.')
sys.path.append('..')
from configs.model_config import encoder_list, pretrained_information
from ranger import Ranger

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

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate of encoder.')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Training epochs.')
    parser.add_argument('--n_steps', type=int, default=3,
                        help='Steps in a iteration.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data.')

    parser.add_argument('--encoder', type=str, choices=['resnet34', 'resnet152', 'poolformer'], default='resnet152',
                        help='The encoder for encoding images into W+ latent codes (resnet152).')
    parser.add_argument('--lambda_e', type=float, default=1,
                        help='Weight of the reconstruction loss.')
    parser.add_argument('--lambda_l', type=float, default=0.8,
                        help='Weight of the LPIPS loss.')

    parser.add_argument('--log_iter', type=int, default=100,
                        help='Print logs every \'log_iter\' iters.')
    parser.add_argument('--output_iter', type=int, default=500,
                        help='Output images every \'output_iter\' iters.')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='Save models every \'save_epoch\' epochs.')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = args.device

    base_dir = f"../experiments/{args.encoder}_{args.name}_steps{args.n_steps}"
    ckpt_dir = f'{base_dir}/ckpt'
    out_dir = f'{base_dir}/out'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{base_dir}/training_config.txt", "wt") as fp:
        for k, v in vars(args).items():
            fp.write(f'{k}: {v}\n')
        fp.close()

    fp = open(f'{base_dir}/training_logs.txt', 'wt')

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

    # Initialize pretrained Encoder
    print(f'Loading pretrained Encoder ...')
    encoder = encoder_list[args.encoder](in_channel=6, size=image_size, n_styles=n_styles).to(device)
    summary(encoder, input_size=(6, 256, 256), batch_size=-1)
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
    train_dataset = set_dataset(args.dataset_type, args.dataset_path, transform,
                                image_size, False, args.train_ratio)
    # test_dataset = set_dataset(args.dataset_type, args.dataset_path, transform,
    #                            image_size, True, args.train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    print('Datasets Loaded.\n')

    # Initialize Adam optimizers
    e_optim = Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.99))

    calculate_lpips_loss = LPIPS(net_type='vgg').to(device)

    # Average w
    w_avg = generator.mapping.w_avg
    # print(w_avg.shape) #(512)
    average_style = w_avg.repeat(n_styles, 1).unsqueeze(0)
    average_image = generator.synthesis(average_style, noise_mode='none').repeat(args.batch_size, 1, 1, 1).detach()

    utils.save_image(average_image, f'{base_dir}/average_image.png', range=(-1, 1), normalize=True)

    # Training
    print('Start Training\n')

    start_time = time.time()
    for epoch_idx in range(1, args.epochs + 1):
        num_iters = len(train_loader)

        for iter_idx, real_image in enumerate(train_loader):
            real_image = real_image.to(device)

            this_batch_size = real_image.shape[0]
            if not this_batch_size == args.batch_size:
                continue
            recovered_image = average_image
            recovered_style = average_style.repeat(args.batch_size, 1, 1)

            recovered_images = []
            encoder.zero_grad()
            for restyle_idx in range(args.n_steps):
                x_input = torch.cat([real_image, recovered_image.clone().detach()], dim=1)
                recovered_style = encoder(x_input) + recovered_style.clone().detach()
                recovered_image = generator.synthesis(recovered_style)

                # Reconstruction Loss
                recon_loss = F.mse_loss(recovered_image, real_image)
                # LPIPS Loss
                lpips_loss = calculate_lpips_loss(recovered_image, real_image)
                # Total loss
                total_loss = recon_loss * args.lambda_e + lpips_loss * args.lambda_l
                total_loss.backward()

                # Record Losses to be logged
                if iter_idx % args.log_iter == 0:
                    training_losses = {
                        'total_loss': total_loss.item(),
                        'recon_loss': recon_loss.item(),
                        'lpips_loss': lpips_loss.item(),
                    }

                # Record images to be output
                if iter_idx % args.output_iter == 0:
                    recovered_images.append(recovered_image)

            # Optimize encoder
            e_optim.step()

            # Print logs
            if iter_idx % args.log_iter == 0:
                used_time = time.time() - start_time
                rest_time = (used_time / ((epoch_idx - 1) * num_iters + iter_idx + 1)) * \
                            ((args.epochs - epoch_idx + 1) * num_iters + num_iters - iter_idx)
                log_output = f'[Training Epoch {epoch_idx}/{args.epochs} Iter {iter_idx:06d}/{num_iters:06d}] ' \
                             f'UsedTime: {time_change(used_time)} RestTime: {time_change(rest_time)} \n' \
                             f"Total_loss: {training_losses['total_loss']:.4f} " \
                             f"Recon_loss: {training_losses['recon_loss']:.4f} " \
                             f"LPIPS_loss: {training_losses['lpips_loss']:.4f} "
                print(log_output, flush=True)
                with open(f'{base_dir}/training_logs.txt', 'a') as fp:
                    fp.write(f'{log_output}\n')

            # Output images
            if iter_idx % args.output_iter == 0:
                saving_image = [real_image]
                for i in range(1, args.n_steps + 1):
                    saving_image.append(recovered_images[args.n_steps - i])
                saving_image = torch.cat(saving_image, dim=-1)
                utils.save_image(saving_image,
                                 f'{out_dir}/Epoch{epoch_idx}_Iter{iter_idx:06d}.png',
                                 normalize=True,
                                 nrow=1,
                                 range=(-1, 1))

        # Save models
        if epoch_idx % args.save_epoch == 0:
            torch.save(encoder.state_dict(), f'{ckpt_dir}/{epoch_idx}.pth')
