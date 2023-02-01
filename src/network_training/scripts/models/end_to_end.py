import torch
from torch import nn
import torch.nn.functional as F

class Dronet(nn.Module):
    """
    Dronet
    with yaw rate appended to the last net
    """
    def __init__(self, input_dim, in_channels=3, extra_dim=0):
        super().__init__()

        # Assume input has the size [in_channels, 64, 64]
        # 1) Very first CNN and MaxPool -> [32, 16, 16]
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.max0 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2) First residual block -> [32, 8, 8]
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0)
        # 3) Second residual block -> [64, 4, 4]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        # 4) Third residual block -> [128, 2, 2]
        self.bn4 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        # 5) Linear layers -> [n_out]
        tmp_dim = int(input_dim / 16 * 256)
        self.reshape = (-1, tmp_dim)
        self.linear0 = nn.Linear(tmp_dim + extra_dim, 1)
        # Init the weight
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.kaiming_normal_(self.conv9.weight)

    def forward(self, x, x_extra=None):
        # 1) Input
        x1 = self.conv0(x)
        x1 = self.max0(x1)
        
        # 2) First residual block
        x2 = self.bn0(x1)
        x2 = F.relu(x2)
        x2 = self.conv1(x2)

        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)

        x1 = self.conv3(x1)
        x3 = torch.add(x1, x2)

        # 3) Second residual block
        x4 = self.bn2(x3)
        x4 = F.relu(x4)
        x4 = self.conv4(x4)

        x4 = self.bn3(x4)
        x4 = F.relu(x4)
        x4 = self.conv5(x4)

        x3 = self.conv6(x3)
        x5 = torch.add(x3, x4)

        # 4) Third residual block
        x6 = self.bn4(x5)
        x6 = F.relu(x6)
        x6 = self.conv7(x6)

        x6 = self.bn5(x6)
        x6 = F.relu(x6)
        x6 = self.conv8(x6)

        x5 = self.conv9(x5)
        x7 = torch.add(x5, x6)

        # 5) Fully-connected layers
        x = torch.flatten(x7)
        x = x.view(self.reshape)
        x = F.relu(x)

        x = F.dropout(x, p=0.5)
        
        if x_extra is not None:
            x = torch.cat((x, x_extra), axis=1)
            output = self.linear0(x)
        else:
            output = self.linear0(x)

        output = torch.tanh(output)

        return output

class Dronet_NoDropout(nn.Module):
    """
    Dronet
    with yaw rate appended to the last net
    """
    def __init__(self, input_dim, in_channels=3, extra_dim=0):
        super().__init__()

        # Assume input has the size [in_channels, 64, 64]
        # 1) Very first CNN and MaxPool -> [32, 16, 16]
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.max0 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2) First residual block -> [32, 8, 8]
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0)
        # 3) Second residual block -> [64, 4, 4]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        # 4) Third residual block -> [128, 2, 2]
        self.bn4 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        # 5) Linear layers -> [n_out]
        tmp_dim = int(input_dim / 16 * 256)
        self.reshape = (-1, tmp_dim)
        self.linear0 = nn.Linear(tmp_dim + extra_dim, 1)
        # Init the weight
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.kaiming_normal_(self.conv9.weight)

    def forward(self, x, x_extra=None):
        # 1) Input
        x1 = self.conv0(x)
        x1 = self.max0(x1)
        
        # 2) First residual block
        x2 = self.bn0(x1)
        x2 = F.relu(x2)
        x2 = self.conv1(x2)

        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)

        x1 = self.conv3(x1)
        x3 = torch.add(x1, x2)

        # 3) Second residual block
        x4 = self.bn2(x3)
        x4 = F.relu(x4)
        x4 = self.conv4(x4)

        x4 = self.bn3(x4)
        x4 = F.relu(x4)
        x4 = self.conv5(x4)

        x3 = self.conv6(x3)
        x5 = torch.add(x3, x4)

        # 4) Third residual block
        x6 = self.bn4(x5)
        x6 = F.relu(x6)
        x6 = self.conv7(x6)

        x6 = self.bn5(x6)
        x6 = F.relu(x6)
        x6 = self.conv8(x6)

        x5 = self.conv9(x5)
        x7 = torch.add(x5, x6)

        # 5) Fully-connected layers
        x = torch.flatten(x7)
        x = x.view(self.reshape)
        x = F.relu(x)
        
        if x_extra is not None:
            x = torch.cat((x, x_extra), axis=1)
            output = self.linear0(x)
        else:
            output = self.linear0(x)

        output = torch.tanh(output)

        return output

class EndToEnd(nn.Module):
    """
    End to End Model
    """
    def __init__(self,
                name,
                input_dim,
                in_channels,
                extra_dim,
                **kwargs):
        super().__init__()
        
        self.name = name
        self.NN = Dronet(input_dim, in_channels, extra_dim)
        self.input_dim = input_dim

    def forward(self, x, x_extra=None):
        output = self.NN(x, x_extra)
        return output

    def loss_function(self, y_pred, y):
        loss = F.mse_loss(y_pred, y, reduction='mean')
        return {'total_loss': loss}

class EndToEnd_NoDropout(nn.Module):
    """
    End to End Model (no dropout)
    """
    def __init__(self,
                name,
                input_dim,
                in_channels,
                extra_dim,
                **kwargs):
        super().__init__()
        
        self.name = name
        self.NN = Dronet_NoDropout(input_dim, in_channels, extra_dim)
        self.input_dim = input_dim

    def forward(self, x, x_extra=None):
        output = self.NN(x, x_extra)
        return output

    def loss_function(self, y_pred, y):
        loss = F.mse_loss(y_pred, y, reduction='mean')
        return {'total_loss': loss}
