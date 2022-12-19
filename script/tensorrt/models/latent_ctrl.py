import torch 
from torch import nn
import torch.nn.functional as F

from .vanilla_vae import Encoder

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


class FC(nn.Module):
    '''
    Simple fully-connected network
    with extra states appended to the first net
    '''
    def __init__(self, z_dim, extra_dim=0):
        super().__init__()
        self.input_layer = nn.Linear(z_dim + extra_dim, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.final_layer = nn.Linear(256, 1)

    def forward(self, x, x_extra):
        # Hidden layers
        if x_extra is not None:
            x = torch.cat((x, x_extra), 1)

        x = self.input_layer(x)
        x = F.relu(x)

        x = self.hidden_layer1(x)
        x = F.relu(x)

        x = self.hidden_layer2(x)
        x = F.relu(x)

        x = self.final_layer(x)
        x = torch.tanh(x)
        
        return x

class LatentCtrl(nn.Module):
    """
    Latent Fully Connected Model
    """
    def __init__(self,
                name,
                z_dim, 
                extra_dim,
                **kwargs):
        super().__init__()
        
        self.name = name
        self.NN = FC(z_dim, extra_dim)
        self.NN.apply(weights_init)
        self.z_dim = z_dim

    def forward(self, x, x_extra=None):
        y_pred = self.NN(x, x_extra)
        return y_pred

    def loss_function(self, y_pred, y):
        loss = F.mse_loss(y_pred, y, reduction='mean')
        return {'total_loss': loss}

    def get_latent_dim(self):
        return self.z_dim


class VAELatentCtrl(nn.Module):
    """
    VAE Encoder + Latent FC Model
    """
    def __init__(self,
                name,
                input_dim,
                in_channels,
                z_dim, 
                extra_dim,
                **kwargs):
        super().__init__()
        
        self.name = name
        self.Encoder = Encoder(input_dim, z_dim, in_channels)
        self.LatentFC = FC(z_dim, extra_dim)
        self.input_dim = input_dim
        self.z_dim = z_dim

    def forward(self, x, z_extra=None):
        mu, _ = self.Encoder(x)
        z = mu # ignore logvar
        y_pred = self.LatentFC(z, z_extra)
        return y_pred

    def get_latent_dim(self):
        return self.z_dim