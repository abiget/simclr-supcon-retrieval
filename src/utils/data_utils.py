import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp

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
            # No fis_trainolders: just load images with label = None
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
    """
    Get a DataLoader for the dataset at data_dir.
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for DataLoader.
        is_train (bool): Whether to apply training transformations or validation (In case).
        image_size (int): Size to which images will be resized.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
    Returns:
        DataLoader: A DataLoader for the dataset.
    """

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

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if is_train else False, num_workers=num_workers)
    return loader


def compute_dataset_statistics(data_dir, image_size=32):
    print(f"Computing dataset statistics for {data_dir}...")
    full_training_data = ImageDataset(root=data_dir, transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]))

    images = torch.stack([image for image, _ in full_training_data], dim=3)
    mean = torch.mean(images)
    std = torch.std(images)

    # Save statistics to file
    torch.save({
        'mean': mean,
        'std': std
    }, os.path.join('dataset_statistics.pth'))

    return mean, std

def load_data_statistics(stat_file, data_dir=None, image_size=32):
    if osp.exists(stat_file):
        stat = torch.load(stat_file)
        mean, std = stat['mean'], stat['std']
    elif data_dir is not None and os.listdir(data_dir):
        # if there is a train dataset, compute statistics
        mean, std = compute_dataset_statistics(data_dir, image_size=image_size)
    else:
        # use imageNet defaults
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    return mean, std


def split_and_organize_dataset(src_dir, dest_dir, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets and physically move files to new directories.
    
    Args:
        src_dir: Source directory containing the dataset
        dest_dir: Destination parent directory where train/ and test/ folders will be created
        test_size: Proportion of the dataset to be used for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_dir, test_dir: Paths to the train and test directories
    """
    # Create destination directories
    train_dir = osp.join(dest_dir, 'train')
    test_dir = osp.join(dest_dir, 'test')
    
    # Make sure destination exists
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load dataset to get sample information
    dataset = ImageDataset(root=src_dir)
    
    if dataset.use_folders:
        # Create class directories in train and test folders
        for class_name in dataset.classes:
            os.makedirs(osp.join(train_dir, class_name), exist_ok=True)
            os.makedirs(osp.join(test_dir, class_name), exist_ok=True)
        
        # Group samples by class
        class_samples = {}
        for path, label in dataset.samples:
            class_name = dataset.classes[label]
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append(path)
        
        # Split and move each class
        for class_name, paths in class_samples.items():
            train_paths, test_paths = train_test_split(
                paths, test_size=test_size, random_state=random_state
            )
            
            # Move training files
            for src_path in train_paths:
                filename = osp.basename(src_path)
                dest_path = osp.join(train_dir, class_name, filename)
                shutil.copy2(src_path, dest_path)
                
            # Move testing files
            for src_path in test_paths:
                filename = osp.basename(src_path)
                dest_path = osp.join(test_dir, class_name, filename)
                shutil.copy2(src_path, dest_path)
                
            print(f"Class {class_name}: {len(train_paths)} training, {len(test_paths)} testing")
            
    else:
        # For unstructured data, just do random split
        paths = [path for path, _ in dataset.samples]
        train_paths, test_paths = train_test_split(
            paths, test_size=test_size, random_state=random_state
        )
        
        # Move training files
        for src_path in train_paths:
            filename = osp.basename(src_path)
            dest_path = osp.join(train_dir, filename)
            shutil.copy2(src_path, dest_path)
            
        # Move testing files
        for src_path in test_paths:
            filename = osp.basename(src_path)
            dest_path = osp.join(test_dir, filename)
            shutil.copy2(src_path, dest_path)
    
    print(f"Dataset split complete!")
    print(f"Training data: {train_dir}")
    print(f"Testing data: {test_dir}")
    
    return train_dir, test_dir