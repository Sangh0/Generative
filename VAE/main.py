import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import VAE
from loss import VAELoss
from utils import plot_loss_graph, plot_generated_images


def train_on_batch(
    device,
    model,
    loss_func,
    optimizer,
    data_loader,
):
    batch_loss = 0
    model.train()
    for idx, (data, _) in enumerate(data_loader):
        data = data.to(device)

        optimizer.zero_grad()
        
        recon_data, mu, log_var = model(data)
        loss = loss_func(data, recon_data, mu, log_var)
        
        loss.backward()
        optimizer.step()
        
        batch_loss += loss.item()
        
        if (idx+1) % 100 == 0:
            print(f'{" "*20} Train Epoch {epoch+1}/{epochs} Batch {idx+1}/{len(data_loader)}')
            print(f'{" "*20} Loss: {loss.item():.3f}')

    return batch_loss / (idx+1)


def test_on_batch(
    device,
    model,
    loss_func,
    data_loader,
):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (data, _) in enumerate(data_loader):
            data = data.to(device)

            recon_data, mu, log_var = model(data)

            loss = loss_func(data, recon_data, mu, log_var)
            test_loss += loss.item()

    return test_loss / (idx + 1)


def training(
    device,
    model,
    loss_func,
    optimizer,
    train_loader,
    test_loader,
    epochs,
):
    train_loss, test_loss = [], []

    for epoch in tqdm(range(epochs)):
        # training
        train_loss = train_on_batch(
            device, model, loss_func, optimizer, train_loader,
        )

        # validation
        test_loss = test_on_batch(
            device, model, loss_func, test_loader,
        )
        
        train_loss.append(train_loss)
        test_loss.append(test_loss)

    return {'model': model, 'train_loss': train_loss, 'test_loss': test_loss}


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset   
    train_set = dsets.MNIST(
        root='./mnist/',
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    test_set = dsets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = VAE(
        in_dim=28* 28,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        z_dim=args.latent_dim,
    ).to(device)

    loss_func = VAELoss(kld_weight=args.kld_weight, recon_weight=args.recon_weight)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    history = training(
        device=device,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
    )

    plot_loss_graph(history)
    plot_generated_images(model)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training VAE', add_help=False)

    # training hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size for efficient batch learning')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--epochs', default=100, type=int,
                        help='epochs')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum constant')
    
    # model hyperparameters
    parser.add_argument('--hidden_dim1', default=512, type=int,
                        help='the hidden dimension of first layer in encoder network')
    parser.add_argument('--hidden_dim2', default=256, type=int,
                        help='the hidden dimension of second layer in encoder network')
    parser.add_argument('--latent_dim', default=2, type=int,
                        help='the dimension of latent vector in encoder network')

    # loss function hyperparameters
    parser.add_argument('--kld_weight', default=1., type=float,
                        help='the balance weight of kld term in loss function')
    parser.add_argument('--recon_weight', default=1., type=float,
                        help='the balance weight of reconstruction term in loss function')
    
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)