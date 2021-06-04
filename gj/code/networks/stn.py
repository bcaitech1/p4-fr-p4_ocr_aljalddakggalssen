import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleSTN(nn.Module):
    """STN that handles arbitary input
    ref:
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    spatial_transformation.py
    """
    def __init__(self, rgb=1):
        super(FlexibleSTN, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=rgb, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # batch_size x 512
        )

        self.fc_loc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
        )

        self.fc_loc2 = nn.Linear(256, 6)

        self.fc_loc2.weight.data.zero_()
        self.fc_loc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))

    def forward(self, x):
        b = x.size(0)
        xs = self.localization(x).view(b, -1)
        xs = self.fc_loc1(xs)
        theta = self.fc_loc2(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = x.to(grid.dtype)
        x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)

        return x