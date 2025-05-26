import torch
import torch.nn as nn
from torchvision import models


class SimCLRBackbone(nn.Module):
    def __init__(self, checkpoint_path=None):
        """
        Initializes the SimCLR backbone with a ResNet-50 encoder.
        Args:
            checkpoint_path (str): Path to the pre-trained simclr model checkpoint.
        """
        super().__init__()
        self.encoder = models.resnet50()
        self.encoder.fc = nn.Identity()  # Remove classification head
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.encoder(x)