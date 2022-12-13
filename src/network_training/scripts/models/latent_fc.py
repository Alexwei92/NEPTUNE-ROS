import torch 
from torch import nn
import torch.nn.functional as F

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


class NN1(nn.Module):
    '''
    Simple fully-connected network
    with extra state appended to the last net
    '''
    def __init__(self, z_dim, extra_dim=0):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 1)
        if extra_dim > 0:
            self.linear5 = nn.Linear(extra_dim + 1, 1)

    def forward(self, x, x_extra):
        # Hidden layers
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        x = self.linear4(x)
        x = torch.tanh(x)

        if x_extra is not None:
            x = torch.cat([x, x_extra], axis=1)
            x = self.linear5(x)
        
        return x

class NN2(nn.Module):
    '''
    Simple fully-connected network
    with yaw rate appended to the first net
    '''
    def __init__(self, z_dim, with_yawRate=False):
        super().__init__()
        if with_yawRate:
            self.input_layer = nn.Linear(z_dim+1, 256)
        else:
            self.input_layer = nn.Linear(z_dim, 256)
        self.hidden_layer1 = nn.Linear(256, 256)
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

class LatentFC(nn.Module):
    """
    Latent Fully Connected Model
    """
    def __init__(self,
                name,
                z_dim, 
                with_yawRate,
                **kwargs):
        super().__init__()
        
        self.name = name
        self.NN = NN2(z_dim, with_yawRate)
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