import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

class Resnet18(nn.Module):
    def __init__(self, input_dim, output_dim=3, n_image=1):
        super().__init__()

        self.n_feature_state = 512 * 4 * 4
      
        resnet18 = models.resnet18(pretrained=False)
        assert resnet18.layer2[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer3[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer4[0].downsample[0].kernel_size == (1, 1)

        resnet18.layer2[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer3[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer4[0].downsample[0].kernel_size = (2, 2)

        new_conv1 = nn.Conv2d(
            n_image * 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet18.conv1 = new_conv1

        self.encoder = torch.nn.Sequential(
            *(list(resnet18.children())[:-2])
        )
        self.last_conv_downsample = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.fc1 = nn.Linear(self.n_feature_state, 256)
        # dist_center_width, rel_angle
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.last_conv_downsample(x)
        x = torch.flatten(x)
        x = x.view(-1, self.n_feature_state)

        x = self.fc1(x)
        x = F.relu(x)

        x1 = self.fc2(x)
        x2 = self.fc3(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.tanh(x)
        return x

class AffordanceNet_Resnet18(nn.Module):
    """
    Affordance Prediction Model
    """
    def __init__(self,
                input_dim,
                output_dim=3,
                n_image=1):
        super().__init__()
        self.net = Resnet18(input_dim=input_dim, output_dim=output_dim, n_image=n_image)

    def forward(self, x):
        return self.net(x)