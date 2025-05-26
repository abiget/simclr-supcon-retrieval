import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    classes = None
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or transforms.ToTensor()

        # Detect if folders exist
        entries = [entry for entry in os.listdir(root) if os.path.isdir(os.path.join(root, entry))]
        if entries:
            # Folders exist: behave like ImageFolder
            self.use_folders = True
            self.classes = sorted(entries)
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
            self.samples = []

            for class_name in self.classes:
                cls_path = os.path.join(root, class_name)
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[class_name]))
        else:
            # No folders: just load images with label = None
            self.use_folders = False
            self.samples = [
                (os.path.join(root, fname), None)
                for fname in os.listdir(root)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        if self.use_folders:
            return image, label
        else:
            return image

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_data_loader(data_dir, batch_size=64, num_workers=4, is_train=True, image_size=32, mean=[0.5]*3, std=[0.5]*3):

    if is_train:
        base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
            ], p=0.8),
            # transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform = TwoCropTransform(base_transform)
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        

    print(f"Loading dataset from {data_dir}...")
    dataset = ImageDataset(root=data_dir, transform=transform)
    print(f"Found {len(dataset)} images {'across ' + str(len(dataset.classes)) + ' classes' if dataset.use_folders  else ''}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader