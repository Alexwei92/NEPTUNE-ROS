import torch
from torch import nn
import torch.nn.functional as F

from .vanilla_vae import Encoder, Decoder

# Weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    GAN Discriminator
    """
    def __init__(self, n_chan=3):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(n_chan,128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        )

        self.last_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512*16*16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        middle_feature = self.convs(x)    
        output = self.last_layer(middle_feature)
        return output.squeeze(), middle_feature.squeeze()


class VAEGAN(nn.Module):
    """
    VAEGAN Model
    """
    def __init__(self,
                name,
                input_dim,
                in_channels,
                z_dim,
                **kwargs):
        super().__init__()

        self.name = name
        self.netE = Encoder(input_dim, z_dim, in_channels)
        self.netG = Decoder(input_dim, z_dim, in_channels)
        self.netD = Discriminator(in_channels)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.z_dim = z_dim
        self.input_dim = input_dim

    def encode(self, x):
        mu, logvar = self.netE(x)
        return mu, logvar

    def reparameterize(self, mu, logvar, with_logvar=True):
        if with_logvar:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x_recon = self.netG(z)
        return x_recon

    def discriminate(self, x):
        label, similarity = self.netD(x)
        return label, similarity

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return [x_recon, x, mu, logvar]

    def sample(self, n_samples, device=torch.device('cuda:0')):
        z = torch.randn(n_samples, self.z_dim).to(device)
        samples = self.decode(z)
        return samples

    def get_latent(self, x, with_logvar=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, with_logvar)
        return z

    def get_latent_dim(self):
        return self.z_dim
