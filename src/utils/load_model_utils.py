from full_model import SupConModel

def load_model(model_path, backbone_only=False, device='cpu', model_type='facenet'):

    if model_type == 'supcon-tuned':
        model = SupConModel.load_fine_tuned_model(
            model_path=model_path, 
            device=device
        )
    elif model_type == 'simclr':
        # Load the model with a SimCLR backbone
        model = SupConModel.from_simclr_pretrained(
            pretrained_path="resnet50-1x.pth", 
            input_dim=2048, 
            proj_dim=256, 
            device=device
        )
    else:
        # Load the model with a FaceNet backbone
        model = SupConModel.from_facenet_pretrained(
            pretrained='vggface2', 
            device=device
        )

    # Return the appropriate component
    if backbone_only:
        return model.backbone
    else:
        return model