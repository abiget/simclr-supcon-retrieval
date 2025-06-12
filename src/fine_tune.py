import os
import requests
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from full_model import SupConModel
from utils.data_utils import get_data_loader
from utils.supconloss import SupConLoss
from utils.data_utils import compute_dataset_statistics


def prepare_supcon_features(features, batch_size):
    """
    Prepare features for SupCon loss by splitting and stacking.
    Args:
        features (torch.Tensor): Features of shape [2 * batch_size, proj_dim].
        batch_size (int): The size of each batch.
    Returns:
        torch.Tensor: Features reshaped to [batch_size, 2, proj_dim].
    """
    f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
    features = torch.stack([f1, f2], dim=1)  # [batch_size, 2, proj_dim]
    return features

def save_model(model, path):
    """
    Save the model state dictionary to a file.
    Args:
        model (nn.Module): The model to save.
        path (str or Path): The file path where the model will be saved.
    """
    torch.save(model.state_dict(), path)

def fine_tune_supcon(model, train_loader, optimizer, criterion, device,
                epochs=100, 
                checkpoint_path="./checkpoints",
                save_every_n_epochs=2,
                scheduler=None
    ):
    """
    Fine-tune the ResNet with SupCon loss.
    Args:
        model (nn.Module): The model to fine-tune.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (str): Device to run the model on ('cpu' or 'cuda').
        epochs (int): Number of epochs to train.
        checkpoint_path (str or Path): Path to save checkpoints.
        save_every_n_epochs (int): Save checkpoint every n epochs.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
    Returns:
        nn.Module: The fine-tuned model.
    """

    for epoch in range(epochs):
        # Training phase
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for step, (images, labels) in enumerate(progress_bar):
            images = torch.cat([images[0], images[1]], dim=0).to(device)
            labels = labels.to(device)
            batch_size = labels.shape[0]

            feats = model(images)
            feats = prepare_supcon_features(feats, batch_size)
            loss = criterion(feats, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1, "step": step})

        # End of epoch
        avg_loss = total_loss / len(train_loader)

        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every_n_epochs == 0 or epoch == (epochs - 1):
            save_model(model, checkpoint_path / f"epoch-{epoch}.pth")

    return model

def init_wandb(project="SupCon-Competition", run_name="supcon_experiment", config=None):
       wandb.init(
        # Set the project where this run will be logged
        project=project,
        # Set a run name
        # (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{run_name}",
        # Track hyperparameters and run metadata
        config=config
    )

def download_simclr_checkpoint(pretrained_path="resnet50-1x.pth"):
    """
    Download the pre-trained SimCLR checkpoint if it does not exist.
    Args:
        pretrained_path (str): Path to save the pre-trained weights file.
    """

    if not os.path.exists(pretrained_path):
        url = "https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x/resolve/main/resnet50-1x.pth"
        print(f"Downloading SimCLR checkpoint from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(pretrained_path, 'wb') as f:
                f.write(response.content)
            print(f"Checkpoint downloaded and saved to {pretrained_path}")
        else:
            raise Exception(f"Failed to download checkpoint. Status code: {response.status_code}")

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Train a SupCon model.")
    parse.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parse.add_argument("--run_name", type=str, default="supcon_experiment_intel_data", help="Name of the run for WandB")
    parse.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save checkpoints")
    parse.add_argument("--pretrained_path", type=str, default="resnet50-1x.pth", help="Path to the pre-trained weights file")
    parse.add_argument("--input_dim", type=int, default=2048, help="Input dimension of the projection head which is the output of resnet50")
    parse.add_argument("--data_dir", type=str, default="data/train", help="Directory containing training data")
    parse.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parse.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parse.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for the optimizer")
    parse.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parse.add_argument("--temperature", type=float, default=0.07, help="Temperature for SupCon loss")
    parse.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"], help="Optimizer to use for training")
    parse.add_argument("--image_size", type=int, default=64, help="Size of the input images (assumed square)")
    parse.add_argument("--proj_dim", type=int, default=256, help="Dimension of the projected features(embeddings)")
    args = parse.parse_args()    

    # ------------------------------------------------
    # Extract arguments
    device = args.device
    input_dim = args.input_dim
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    checkpoint_path = args.checkpoint_path
    run_name = args.run_name
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    temperature = args.temperature
    optimizer = args.optimizer
    pretrained_path = args.pretrained_path
    proj_dim = args.proj_dim
    # ------------------------------------------------
    # print all the parameters in dictionary format
    params = {
        "device": device,
        "input_dim": input_dim,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "epochs": epochs,
        "checkpoint_path": checkpoint_path,
        "run_name": run_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "optimizer": optimizer,
        "image_size": args.image_size,
        "pretrained_path": pretrained_path,
        "proj_dim": proj_dim
    }
    print("Fine-tuning parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    # down load the resnet50 checkpoint if not already present
    download_simclr_checkpoint(pretrained_path=pretrained_path)

    # compute dataset statistics
    mean, std = compute_dataset_statistics(data_dir, image_size=args.image_size)

    # train_loader = get_dummy_loader(batch_size=10) # For testing purposes, replace with actual data loader
    train_loader = get_data_loader(data_dir, batch_size=batch_size, is_train=True, image_size=args.image_size, mean=mean, std=std)

    config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "temperature": temperature
        }
    
    # WandB initialization
    init_wandb(project="SupCon-Competition", run_name=run_name, config=config)

    # Create the checkpoint directory
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path = checkpoint_path / run_name
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Create the model using the factory method
    model = SupConModel.from_resnet_checkpoint(
        pretrained_path=pretrained_path,
        input_dim=input_dim,
        proj_dim=proj_dim,
        device=device,
    )

    criterion = SupConLoss(temperature=temperature)

    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Add learning rate scheduler - Cosine Annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    model = fine_tune_supcon(model, train_loader, optimizer, criterion, device, epochs=epochs, checkpoint_path=checkpoint_path, scheduler=scheduler)
    # Save the final model
    save_model(model, checkpoint_path / "supcon_model_final.pth")
    wandb.finish()