import os
import requests
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, DataLoader
from models.backbone import SimCLRBackbone
from models.projection import ProjectionHead
from models.full_model import SupConModel
from utils.data_utils import get_data_loader
from utils.supconloss import SupConLoss
from utils.data_utils import compute_dataset_statistics

# Dummy dataset for testing purposes-----------------------------------
class DummySupConDataset(Dataset):
    def __init__(self, num_samples=100, channels=3, height=32, width=32, num_classes=10):
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create two random augmentations
        img1 = torch.randn(self.channels, self.height, self.width)
        img2 = torch.randn(self.channels, self.height, self.width)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return (img1, img2), label

def get_dummy_loader(batch_size=16):
    dataset = DummySupConDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
# ---------------------------------------------------------------------

def prepare_supcon_features(features, batch_size):
    f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
    features = torch.stack([f1, f2], dim=1)  # [batch_size, 2, proj_dim]
    return features

def save_model(model, path):
    torch.save(model.state_dict(), path)

def train_supcon(model, train_loader, optimizer, criterion, device,
                epochs=100, 
                checkpoint_path="./checkpoints",
                save_every_n_epochs=2
    ):
            
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

def download_simclr_checkpoint(checkpoint_path="resnet50-1x.pth"):

    if not os.path.exists(checkpoint_path):
        url = "https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x/resolve/main/resnet50-1x.pth"
        print(f"Downloading SimCLR checkpoint from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(checkpoint_path, 'wb') as f:
                f.write(response.content)
            print(f"Checkpoint downloaded and saved to {checkpoint_path}")
        else:
            raise Exception(f"Failed to download checkpoint. Status code: {response.status_code}")

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Train a SupCon model.")
    parse.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parse.add_argument("--run_name", type=str, default="supcon_experiment", help="Name of the run for WandB")
    parse.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save checkpoints")
    parse.add_argument("--input_dim", type=int, default=2048, help="Input dimension of the projection head")
    parse.add_argument("--data_dir", type=str, default="data/train", help="Directory containing training data")
    parse.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parse.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parse.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parse.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parse.add_argument("--temperature", type=float, default=0.07, help="Temperature for SupCon loss")
    parse.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer to use for training")
    parse.add_argument("--image_size", type=int, default=32, help="Size of the input images (assumed square)")
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
    # ------------------------------------------------

    # down load the resnet50 checkpoint if not already present
    download_simclr_checkpoint(checkpoint_path="resnet50-1x.pth")

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
        checkpoint_path="resnet50-1x.pth",
        input_dim=input_dim,
        proj_dim=128,
        device=device,
    )

    criterion = SupConLoss(temperature=temperature)

    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model = train_supcon(model, train_loader, optimizer, criterion, device, epochs=epochs, checkpoint_path=checkpoint_path)
    # Save the final model
    save_model(model, checkpoint_path / "supcon_model_final.pth")
    wandb.finish()