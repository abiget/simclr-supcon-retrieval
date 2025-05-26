from models.full_model import SupConModel

def load_model(model_path, backbone_only=False, device='cpu'):
    # Use the classmethod to load the full model
    
    model = SupConModel.load_trained_model(
        model_path=model_path, 
        device=device
    )
    
    # Return the appropriate component
    if backbone_only:
        return model.backbone
    else:
        return model