import torch
import torch.nn as nn


def sinusoidal_embedding(n, dim):
    emb = torch.zeros(n, dim)
    wk = torch.tensor([1 / 10000 ** (2 * j / dim) for j in range(dim)])
    wk = wk.reshape((1, dim))
    t = torch.arange(n).reshape((n, 1))
    emb[:, ::2] = torch.sin(t * wk[:, ::2])
    emb[:, 1::2] = torch.cos(t * wk[:, ::2])
    return emb


class DDPM(nn.Module):
    
    def __init__(
        self,
        network,
        n_steps=200,
        min_beta=1e-4,
        max_beta=0.02,
        device=None,
        img_shape=(1,28,28),
    ):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.img_shape = img_shape
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        
    def forward(self, x0, t, eps=None):
        N, C, H, W = x0.shape
        alpha_bar = self.alphas_bar[t]

        if eps is None:
            eps = torch.randn(N, C, H, W).to(self.device)

        noise = alpha_bar.sqrt().reshape(N, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(N, 1, 1, 1) * eps
        return noise

    def backward(self, x, t):
        return self.network(x, t)


class UNetBlock(nn.Module):

    def __init__(
        self,
        shape,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=None,
        normalize=True,
    ):
        super(UNetBlock, self).__init__()
        self.norm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        x = self.norm(x) if self.normalize else x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class TimeEmbBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(TimeEmbBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        self.time_emb = nn.Embedding(n_steps, time_emb_dim)
        self.time_emb.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_emb.requires_grad_(False)

        self.tb1 = TimeEmbBlock(time_emb_dim, 1)
        self.cb1 = nn.Sequential(
            UNetBlock((1, 28, 28), in_dim=1, out_dim=10),
            UNetBlock((10, 28, 28), in_dim=10, out_dim=10),
            UNetBlock((10, 28, 28), in_dim=10, out_dim=10),
        )
        self.down1 = nn.Conv2d(10, 10, kernel_size=4, stride=2, padding=1)

        self.tb2 = TimeEmbBlock(time_emb_dim, 10)
        self.cb2 = nn.Sequential(
            UNetBlock((10, 14, 14), in_dim=10, out_dim=20),
            UNetBlock((20, 14, 14), in_dim=20, out_dim=20),
            UNetBlock((20, 14, 14), in_dim=20, out_dim=20),
        )
        self.down2 = nn.Conv2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.tb3 = TimeEmbBlock(time_emb_dim, 20)
        self.cb3 = nn.Sequential(
            UNetBlock((20, 7, 7), in_dim=20, out_dim=40),
            UNetBlock((40, 7, 7), in_dim=40, out_dim=40),
            UNetBlock((40, 7, 7), in_dim=40, out_dim=40),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=2, stride=1),
            nn.SiLU(),
            nn.Conv2d(40, 40, kernel_size=4, stride=2, padding=1),
        )

        self.tb_bottleneck = TimeEmbBlock(time_emb_dim, 40)
        self.cb_bottleneck = nn.Sequential(
            UNetBlock((40, 3, 3), in_dim=40, out_dim=20),
            UNetBlock((20, 3, 3), in_dim=20, out_dim=20),
            UNetBlock((20, 3, 3), in_dim=20, out_dim=40),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, kernel_size=2, stride=1),
        )
        self.tb4 = TimeEmbBlock(time_emb_dim, 80)
        self.cb4 = nn.Sequential(
            UNetBlock((80, 7, 7), in_dim=80, out_dim=40),
            UNetBlock((40, 7, 7), in_dim=40, out_dim=20),
            UNetBlock((20, 7, 7), in_dim=20, out_dim=20),
        )

        self.up2 = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)
        self.tb5 = TimeEmbBlock(time_emb_dim, 40)
        self.cb5 = nn.Sequential(
            UNetBlock((40, 14, 14), in_dim=40, out_dim=20),
            UNetBlock((20, 14, 14), in_dim=20, out_dim=10),
            UNetBlock((10, 14, 14), in_dim=10, out_dim=10),
        )

        self.up3 = nn.ConvTranspose2d(10, 10, kernel_size=4, stride=2, padding=1)
        self.tb_out = TimeEmbBlock(time_emb_dim, 20)
        self.cb_out = nn.Sequential(
            UNetBlock((20, 28, 28), in_dim=20, out_dim=10),
            UNetBlock((10, 28, 28), in_dim=10, out_dim=10),
            UNetBlock((10, 28, 28), in_dim=10, out_dim=10, normalize=False),
        )

        self.conv_out = nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t = self.time_emb(t)
        n = len(x)

        out1 = self.cb1(x + self.tb1(t).reshape(n, -1, 1, 1))
        out2 = self.cb2(self.down1(out1) + self.tb2(t).reshape(n, -1, 1, 1))
        out3 = self.cb3(self.down2(out2) + self.tb3(t).reshape(n, -1, 1, 1))

        bottleneck = self.cb_bottleneck(self.down3(out3) + self.tb_bottleneck(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(bottleneck)), dim=1)
        out4 = self.cb4(out4 + self.tb4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.cb5(out5 + self.tb5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.cb_out(out + self.tb_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)
        return out