import torch
from torch import nn
from torch.nn import functional as F
from torchvision import utils

from argparse import ArgumentParser
import math
import os

import time
from mytime import time_change

import sys

sys.path.append('.')
sys.path.append('..')
from configs.model_config import encoder_list, pretrained_information
from ranger import Ranger
import dnnlib
import legacy

from criteria.lpips.lpips import LPIPS

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name of pretrained StyleGAN2 (e.g. FFHQ256).')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate of encoder.')
    parser.add_argument('--iters', type=int, default=100000,
                        help='Training iterations.')
    parser.add_argument('--n_steps', type=int, default=5,
                        help='Steps in a iteration.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--encoder', type=str, choices=['resnet152'], default='resnet152',
                        help='The encoder for encoding images into W+ latent codes (resnet152).')
    parser.add_argument('--lambda_recon', type=float, default=1,
                        help='Weight of the reconstruction loss.')
    parser.add_argument('--lambda_lpips', type=float, default=0.8,
                        help='Weight of the LPIPS loss.')
    parser.add_argument('--lambda_recov', type=float, default=10,
                        help='Weight of the recovery loss.')
    parser.add_argument('--log_iter', type=int, default=100,
                        help='Print logs every \'log_iter\' iters.')
    parser.add_argument('--output_iter', type=int, default=200,
                        help='Output images every \'output_iter\' iters.')
    parser.add_argument('--save_iter', type=int, default=50000,
                        help='Save models every \'save_iter\' iters.')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = args.device

    base_dir = f"../experiments/Restyle_Encoder_only/{args.encoder}_{args.name}_steps{args.n_steps}"
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
    #with open(pretrained_path, 'rb') as f:
    #    generator = pickle.load(f)['G_ema'].to(device)
    generator.train()
    print(f'pretrained StyleGAN2 ({args.name}) on {device} loaded')

    # Initialize pretrained Encoder
    print(f'\nLoading pretrained Encoder ...')
    encoder = encoder_list[args.encoder](in_channel=6, size=image_size, n_styles=n_styles).to(device)
    encoder.train()
    print(f'Pretrained Encoder loaded.\n')

    # Initialize Adam optimizers
    e_optim = Ranger(encoder.parameters(), lr=args.lr)

    calculate_lpips_loss = LPIPS(net_type='vgg').to(device).eval()

    batch_size = args.batch_size

    # Average w
    w_avg = generator.mapping.w_avg
    # print(w_avg.shape) #(512)
    average_style = w_avg.repeat(n_styles, 1).unsqueeze(0)
    average_image = generator.synthesis(average_style, noise_mode='none').repeat(batch_size, 1, 1, 1).detach()

    utils.save_image(average_image,
                     f'{base_dir}/average_image.png',
                     range=(-1, 1),
                     normalize=True)

    # Training
    print('Start Training\n')

    start_time = time.time()
    for iter_idx in range(1, args.iters + 1):
        with torch.no_grad():
            z = torch.randn(batch_size, generator.z_dim).to(device)
            origin_style = generator.mapping(z, None)
            container_image = generator.synthesis(origin_style, noise_mode='none')

        recovered_images = []
        encoder.zero_grad()
        for restyle_idx in range(args.n_steps):
            if restyle_idx == 0:
                x_input = torch.cat([container_image, average_image], dim=1).clone().detach().requires_grad_(True)
                recovered_style = encoder(x_input) + average_style
                recovered_image = generator.synthesis(recovered_style)
            else:
                x_input = torch.cat([container_image, recovered_image.clone().detach().requires_grad_(True)],
                                    dim=1).clone().detach().requires_grad_(True)
                recovered_style = encoder(x_input) + recovered_style.clone().detach().requires_grad_(True)
                recovered_image = generator.synthesis(recovered_style)

            # Reconstruction Loss
            recon_loss = F.mse_loss(recovered_image, container_image)
            # LPIPS Loss
            lpips_loss = calculate_lpips_loss(recovered_image, container_image)
            # Recovery Loss
            # print(recovered_style.shape) # [1,14,512]
            recov_loss = nn.functional.l1_loss(recovered_style[:,:7], origin_style[:,:7])
            # Total loss
            total_loss = recon_loss * args.lambda_recon + lpips_loss * args.lambda_lpips + recov_loss * args.lambda_recov
            total_loss.backward()

            # Record images to be output
            if iter_idx % args.output_iter == 0:
                recovered_images.append(recovered_image)

        # Optimize encoder
        e_optim.step()

        # Print logs
        if iter_idx % args.log_iter == 0:
            # Recovered loss of style codes
            recov_loss = torch.mean(torch.abs(recovered_style - origin_style))

            log_output = f'[{iter_idx:06d}/{args.iters:06d}] ' \
                         f'UsedTime: {time_change(time.time() - start_time)} RestTime: {time_change((time.time() - start_time) / iter_idx * (args.iters - iter_idx))} \n' \
                         f'Total_loss: {total_loss.item():.4f} ' \
                         f'Recon_loss: {recon_loss.item():.4f} ' \
                         f'LPIPS_loss: {lpips_loss.item():.4f} ' \
                         f'Recov_loss: {recov_loss:.4f} '

            print(log_output)
            with open(f'{base_dir}/training_logs.txt', 'a') as fp:
                fp.write(f'{log_output}\n')

        # Output images
        if iter_idx % args.output_iter == 0:
            saving_image = [container_image[0]]
            for i in range(1, args.n_steps + 1):
                saving_image.append(recovered_images[args.n_steps - i][0])
            saving_image = torch.stack(saving_image, dim=0)
            utils.save_image(saving_image,
                             f'{out_dir}/EncodeSample_{iter_idx:06d}.png',
                             normalize=True,
                             range=(-1, 1))

        # Save models
        if iter_idx % args.save_iter == 0:
            torch.save(encoder.state_dict(), f'{ckpt_dir}/{iter_idx}.pth')
