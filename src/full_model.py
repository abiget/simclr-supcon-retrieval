import torch
from torch import nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1

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
            ckpt = torch.load(checkpoint_path, map_location='cpu' if not torch.cuda.is_available() else 'cuda:0')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.encoder(x)

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

class SupConModel(nn.Module):
    """
    Full model for supervised contrastive learning.
    Combines a backbone network and a projection head.
    Args:
        backbone (nn.Module): Backbone network for feature extraction.
        proj_head (nn.Module): Projection head for mapping features to a lower-dimensional space.
    """
    def __init__(self, backbone, proj_head=None):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head

    def forward(self, x):
        h = self.backbone(x)
        # apply the projection head only if the model is not pre-trained(facenet)
        if self.proj_head is not None:
            return self.proj_head(h)
        return h

    @classmethod
    def from_resnet_checkpoint(cls, pretrained_path, input_dim=2048, proj_dim=256, device="cpu"):
        """Factory method to create a pre-trained model with simclr trained resnet50 and ready for fine-tune it.
        Args:
            pretrained_path (str): Path to the pre-trained ResNet weights.
            input_dim (int): Input dimension of the projection head.
            proj_dim (int): Output dimension of the projection head.
            device (str): Device to load the model on ('cpu' or 'cuda').
        Returns:
            SupConModel: An instance of the SupConModel with a ResNet backbone.
        """
        backbone = SimCLRBackbone(checkpoint_path=pretrained_path)
        proj_head = ProjectionHead(input_dim=input_dim, proj_dim=proj_dim)
        model = cls(backbone, proj_head)
        return model.to(device)
    
    @classmethod
    def from_simclr_pretrained(cls, pretrained_path='resnet50-1x.pth', input_dim=2048, proj_dim=256, device="cpu"):
        """
        Factory method to create a model with a SimCLR backbone.
        This uses the pre-trained weights from the specified path.
        Args:
            pretrained_path (str): Path to the pre-trained SimCLR weights.
            input_dim (int): Input dimension of the projection head.
            proj_dim (int): Output dimension of the projection head.
            device (str): Device to load the model on ('cpu' or 'cuda').
        Returns:
            SupConModel: An instance of the SupConModel with a SimCLR backbone.
        """
        backbone = SimCLRBackbone(checkpoint_path=pretrained_path)
        model = cls(backbone=backbone)
        return model.to(device)

    @classmethod
    def from_facenet_pretrained(cls, pretrained='vggface2', device="cpu"):
        """
        Factory method to create a model with a FaceNet backbone.
        This uses the pretrained weights from the specified source.
        Args:
            pretrained (str): Pretrained model source, e.g., 'vggface2'.
            device (str): Device to load the model on ('cpu' or 'cuda').
        Returns:
            SupConModel: An instance of the SupConModel with a FaceNet backbone.
        """
        backbone = InceptionResnetV1(pretrained=pretrained).eval()
        model = cls(backbone=backbone)
        return model.to(device)

    @classmethod
    def load_fine_tuned_model(cls, model_path, input_dim=2048, proj_dim=256, device="cpu"):
        """
        Load a fine tuned SupConModel (resnet50) from a saved checkpoint.
        This loads both backbone and projection head weights.
        Args:
            model_path (str): Path to the saved model checkpoint.
            input_dim (int): Input dimension of the projection head.
            proj_dim (int): Output dimension of the projection head.
            device (str): Device to load the model on ('cpu' or 'cuda').
        Returns:
            SupConModel: An instance of the SupConModel with fine-tuned weights.
        """
        # First, create the model structure
        backbone = SimCLRBackbone()  
        proj_head = ProjectionHead(input_dim=input_dim, proj_dim=proj_dim)
        model = cls(backbone=backbone, proj_head=proj_head)

        # Then load the state dictionary
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        
        return model.to(device)