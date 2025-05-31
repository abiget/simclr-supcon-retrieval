import torch
from torch import nn

class ProjectionHead(nn.Module):
    """
    Projection head for SimCLR, maps features to a lower-dimensional space.
    Args:
        input_dim (int): Dimension of the input features (output of SimCLR).
        proj_dim (int): Dimension of the projected features.
    """
    def __init__(self, input_dim=4096, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
    def forward(self, x):
        return nn.functional.normalize(self.mlp(x), dim=1)