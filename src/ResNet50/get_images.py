'''This file contains function necessary for all the models; 
   such as loading the images from the gallery and query folders, 
   passing them through the networks to extract embeddings, 
   and finally comparing these embeddings to return the most similar images as .json
   '''

import os
import json
import requests
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Image transformation for ResNet
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Custom dataset for a folder of images


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(image_path)
        return image, filename

# Create a DataLoader from an image folder


def get_image_loader(image_folder, batch_size=BATCH_SIZE):
    dataset = ImageFolderDataset(image_folder, transform=image_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


# Extract embeddings from a model for all images in a DataLoader
'''The extract_embeddings function passes images through the model, gets the 2048-length feature vector for each image and saves it into a dictionary. The dictionary has the following structure {filename : vector}.

Since we are not training the model, but we are in inference mode (model.eval()) (i.e., we are not updating the model weights to improve accuracy), we use torch.no_grad to tell PyTorch not to track gradients, which are needed only for training, making the code faster and using less memory. In this case we can do this because all we're doing is:
1. Sending an image through the pretrained network,
2. Getting the feature vector (2048-d) from the second-last layer
3. Saving that vector and not using it to calculate loss or update the model.'''


def extract_embeddings(dataloader, model, device):
    model.to(device)
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for images, filenames in tqdm(dataloader):
            images = images.to(device)
            features = model(images)  # shape: (batch, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # shape: (batch, 2048)
            for fname, vec in zip(filenames, features):
                embeddings[fname] = vec.cpu().numpy()
    return embeddings


# Compute top-k most similar images using cosine similarity
'''get_top_k_similar compares every query image to all gallery images using cosine similarity adn returns the filenames of the top-k most similar images.
Cosine similarity tells how "directionally" close two vectors are and is perfect for comparing features. (need to cite a research in the report explaining why)
'''


def get_top_k_similar(query_embeddings, gallery_embeddings, k=10):
    gallery_filenames = list(gallery_embeddings.keys())
    gallery_matrix = np.stack([gallery_embeddings[fn]
                              for fn in gallery_filenames])

    results = {}
    for query_fname, query_vec in query_embeddings.items():
        query_vec = query_vec.reshape(1, -1)
        similarities = cosine_similarity(query_vec, gallery_matrix)[0]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_filenames = [gallery_filenames[i] for i in top_k_indices]
        results[query_fname] = top_k_filenames
    return results

# Save results to a JSON file


def save_results_to_json(results, output_path="submission.json"):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_path}")


def submit(results, groupname="beasts", url="http://65.108.245.177:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
