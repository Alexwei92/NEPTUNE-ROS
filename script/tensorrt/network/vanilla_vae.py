import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    VAE Encoder from image size
    """
    def __init__(self, input_dim, n_out, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.reshape = (-1, int(512*(input_dim/32)**2))
        self.linear0 = nn.Linear(int(512*(input_dim/32)**2), n_out)
        self.linear1 = nn.Linear(int(512*(input_dim/32)**2), n_out)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        x = x.view(self.reshape)
        mu = self.linear0(x) # mean
        logvar = self.linear1(x) # log(variance)
        
        return mu, logvar

class Decoder(nn.Module):
    """
    VAE Decoder to image size
    """
    def __init__(self, output_dim, n_in, in_channels=3):
        super().__init__()
        
        # 1) Linear layers
        self.linear0 = nn.Linear(n_in, int(512*(output_dim/32)**2))
        self.reshape = (-1, 512, int(output_dim/32), int(output_dim/32))

        # 2) Transposed convolutional layers
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 1) Linear layer3
        x = self.linear0(x)
        x = x.view(self.reshape)

        # 2) Transposed convolutional layers
        x = self.deconv(x)

        return x

class VanillaVAE(nn.Module):
    """
    Vanilla VAE Model
    """
    def __init__(self,
                input_dim,
                in_channels,
                z_dim):
        super().__init__()
        
        self.Encoder = Encoder(input_dim, z_dim, in_channels)
        self.Decoder = Decoder(input_dim, z_dim, in_channels)
        self.z_dim = z_dim
        self.input_dim = input_dim

    def encode(self, x):
        mu, logvar = self.Encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar, with_logvar=True):
        if with_logvar:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x_recon = self.Decoder(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return [x_recon, x, mu, logvar]

    def loss_function(self, *args, **kwargs):
        x_recon = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]
        
        batch_size = x_recon.size(0)
        mse = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)        
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # kld_z = kld.mean(0)
        kld = kld.sum(1).mean(0)
        total_loss = mse + kld
        return {'total_loss': total_loss,
                'mse_loss': mse,
                'kld_loss': kld,
                # 'kld_loss_z': kld_z
            }

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