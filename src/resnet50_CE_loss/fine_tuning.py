import os
import time
import torch
from collections import Counter
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os

# Get the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your local train folder inside the project folder
TRAIN_DIR = os.path.join(CURRENT_DIR, "data", "train")

# Path to save the fine-tuned model (same folder as the script)
OUTPUT_PATH = os.path.join(CURRENT_DIR, "fine_tuned_model.pth")


# Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
print("📂 Loading training data...")
# datasets.ImageFolder() creates a labeled dataset from a folder of images.
# Scans subfolders inside TRAIN_DIR assuming each subfolder name is a class label
# Assigns a numerical label to each class (class1 = 0, class2 = 1)
# Returns one sample at a time applying train_transform
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

# Check how many images there are per class
labels = [sample[1] for sample in train_dataset.samples]
class_counts = Counter(labels)

print("📊 Class distribution:")
for class_idx, count in class_counts.items():
    print(f"Class {train_dataset.classes[class_idx]}: {count} images")

# split the dataset into training and validation subsets
val_ratio = 0.2
val_size = int(len(train_dataset) * val_ratio)
train_size = len(train_dataset) - val_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# DataLoader feeds train_dataset to your model in mini-batches
# (at each iteration it returns a batch of images of size [BATCH_SIZE, 3, 224, 224] + a tensor of the respective classes labels)
# Shuffle the order of samples each epoch (if shuffle=True)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained model and modify classifier
print("🧠 Loading pretrained ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features

# # Freeze all pretrained layers
# for param in model.parameters():
#     param.requires_grad = False

# Replace final layer (classifier) with a new trainable one matching number of classes
model.fc = nn.Linear(num_features, len(train_dataset.classes))

# # Make sure classifier is trainable
# for param in model.fc.parameters():
#     param.requires_grad = True

# move model to GPU
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# adam is an optimizer that updates model weights using gradients
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # train_loader returns batches of images and labels
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Clears old gradients from the previous step. Always required before loss.backward() to avoid accumulation
        optimizer.zero_grad()
        # The model processes the input and produces class scores for each image
        # outputs shape: [BATCH_SIZE, num_classes] beacuse the model outputs one logit for each class
        outputs = model(images)
        # Calculate the loss between predicted scores and true labels
        # Produces one scalar value for the current batch
        loss = criterion(outputs, labels)
        # Calculates gradients of the loss with respect to model weights
        loss.backward()
        # Updates the model weights using the gradients
        optimizer.step()
        # Accumulate loss for reporting (weighted by the dimension of the batch)
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


best_val_loss = float("inf")
patience = 3  # Early stopping patience
counter = 0
train_losses = []
val_losses = []
# Training loop
start = time.time()
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), OUTPUT_PATH)
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping.")
            break

end = time.time()
print(f"⏱️ Training completed in {(end - start)/60:.2f} minutes")
print(f"✅ Best validation loss: {best_val_loss:.4f}")
print(f"🧠 Best model saved to: {OUTPUT_PATH}")

# Plot training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")

# inspect model weights
model_dict = torch.load(OUTPUT_PATH,
                        map_location='cpu', weights_only=True)
print(f"Model loaded with {len(model_dict)} parameters.")
