import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader

from models import DDPM, UNet
from utils import show_images, generate_new_images


def training(ddpm, data_loader, epochs, optimizer, device, display=False, store_path='./ddpm_best.pt'):
    loss_func = nn.MSELoss()
    n_steps = ddpm.n_steps
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc='Training', colour='#00ff00'):
        epoch_loss = 0

        for batch, data in enumerate(tqdm(data_loader, leave=False, desc=f'Epoch {epoch+1}/{epochs}', colour='#005500')):
            x0 = data[0].to(device)
            n = len(x0)

            eps = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            noise_images = ddpm(x0, t, eps)

            eps_theta = ddpm.backward(noise_images, t.reshape(n, -1))

            loss = loss_func(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * n / len(data_loader)

        if display:
            show_images(generate_new_images(ddpm, device=device))
        
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            print(f'Saved weight, loss decreased {best_loss:.3f}')

        print(f'Loss: {epoch_loss:.3f}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Diffusion Model Training', add_help=False)

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'],
                        help='select dataset for training diffusion model')
    parser.add_argument('--store_path', type=str, default='./ddpm_best.pt',
                        help='a directory to save weight file')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=30,
                        help='total epochs for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    
    # model parameters
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='number of steps')
    parser.add_argument('--time_emb_dim', type=int, default=100,
                        help='the dimension of time embedding layer')
    parser.add_argument('--min_beta', type=float, default=1e-4,
                        help='min beta')
    parser.add_argument('--max_beta', type=float, default=0.02,
                        help='max beta')

    return parser


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2),
    ])

    if args.dataset == 'mnist':
        dset_FN = dsets.MNIST
    
    else:
        dset_FN = dsets.FashionMNIST

    dataset = dset_FN(
        root='./dataset',
        train=True,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet(n_steps=args.n_steps, time_emb_dim=args.time_emb_dim)
    ddpm = DDPM(
        network=unet,
        n_steps=args.n_steps,
        min_beta=args.min_beta,
        max_beta=args.max_beta,
        device=device,
    )

    optimizer = optim.Adam(
        ddpm.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    training(
        ddpm,
        train_loader,
        args.epochs,
        optimizer,
        device,
        display=True,
        store_path=args.store_path,
    )

    best_model = ddpm
    best_model.load_state_dict(torch.load(args.store_path, map_location=device))
    best_model.eval()

    generated = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name='generated.gif',
    )

    show_images(generated, 'Generated Images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Diffusion Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)