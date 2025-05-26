import torch
from torch import nn
from .backbone import SimCLRBackbone
from .projection import ProjectionHead

class SupConModel(nn.Module):
    """
    Full model for supervised contrastive learning.
    Combines a backbone network and a projection head.
    Args:
        backbone (nn.Module): Backbone network for feature extraction.
        proj_head (nn.Module): Projection head for mapping features to a lower-dimensional space.
    """
    def __init__(self, backbone, proj_head):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head
    def forward(self, x):
        h = self.backbone(x)
        z = self.proj_head(h)
        return z

    @classmethod
    def from_resnet_checkpoint(cls, checkpoint_path, input_dim=2048, proj_dim=128, device="cpu"):
        """Factory method to create a pre-trained model"""
        backbone = SimCLRBackbone(checkpoint_path=checkpoint_path)
        proj_head = ProjectionHead(input_dim=input_dim, proj_dim=proj_dim)
        model = cls(backbone, proj_head)
        return model.to(device)
    
    @classmethod
    def load_trained_model(cls, model_path, input_dim=2048, proj_dim=128, device="cpu"):
        """
        Load a fully trained SupConModel from a saved checkpoint.
        This loads both backbone and projection head weights.
        """
        # First, create the model structure
        backbone = SimCLRBackbone()  
        proj_head = ProjectionHead(input_dim=input_dim, proj_dim=proj_dim)
        model = cls(backbone, proj_head)
        
        # Then load the state dictionary
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        return model.to(device)