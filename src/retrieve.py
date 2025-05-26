import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from utils.plot_utils import plot_query_and_similars
from utils.data_utils import get_data_loader
from utils.load_model_utils import load_model
from torch.utils.data import DataLoader, Dataset

def extract_backbone_features(model, images, device='cpu'):
    """Extract features from the backbone (encoder) only."""
    model.eval()
    with torch.no_grad():
        # If model is SupConModel, get backbone
        if hasattr(model, "backbone"):
            features = model.backbone(images.to(device))
        else:
            features = model(images.to(device))
    return features.cpu().numpy()

def evaluation_transform(img_size=32, mean=[0.5]*3, std=[0.5]*3):
    """Transform for evaluation, resizing and normalizing images."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def normalize_feature(feature):
    return feature / np.linalg.norm(feature, axis=1, keepdims=True)

def precompute_dataset_embeddings(model, dataset, device='cpu'):
    """
    Precompute embeddings for all images in the dataset.
    Returns features array and corresponding paths.
    """
    model.eval()
    model = model.to(device)

    transform = evaluation_transform()

    all_features = []
    all_paths = []

    for img_path, _ in tqdm(dataset.samples, desc="Precomputing embeddings"):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        feat = extract_backbone_features(model, img_tensor, device)
        # Normalize feature
        feat = normalize_feature(feat)
        all_features.append(feat)
        all_paths.append(img_path)
    
    all_features = np.vstack(all_features)
    
    return all_features, all_paths

def find_similar_images_with_precomputed(query_feature, all_features, all_paths, top_k=5):
    """
    Find most similar images using precomputed features.
    """
    # Compute cosine similarity since normalized features are used
    similarities = np.dot(all_features, query_feature.T).flatten()

    # Get top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    similar_images = [(all_paths[i], similarities[i]) for i in top_indices]

    return similar_images

def get_query_embedding(model, image_path, device='cpu'):
    """
    Get embedding for a query image.
    """
    model.eval()
    model = model.to(device)
    
    transform = evaluation_transform()
    
    # Extract feature for the query image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    query_feature = extract_backbone_features(model, image_tensor, device)
    # Normalize query feature
    query_feature = normalize_feature(query_feature)
    
    return query_feature


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description="Image Retrieval using SupCon Model")
    argparser.add_argument("--model_path", type=str, default="checkpoints/supcon_experiment/supcon_model_final.pth", help="Path to the trained model")
    argparser.add_argument("--gallery_dir", type=str, default="test/", help="Directory containing gallery images")
    argparser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader")
    argparser.add_argument("--test_dir", type=str, default="test/", help="Directory containing test images for retrieval")
    argparser.add_argument("--top_k", type=int, default=5, help="Number of top similar images to retrieve")
    argparser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    argparser.add_argument("--output_file", type=str, default="submission.json", help="Path to save the output JSON file")
    args = argparser.parse_args()
    # Load the model
    device = args.device

    model = load_model(args.model_path, device=device, backbone_only=True)

    gallery_loader = get_data_loader(
        data_dir=args.gallery_dir,
        batch_size=args.batch_size,
        num_workers=4,
        is_train=False,
    )

    # Precompute dataset embeddings 
    dataset_features, dataset_paths = precompute_dataset_embeddings(model, gallery_loader.dataset, device=device)

    # # Process multiple query images efficiently
    query_img_paths = [os.path.join(args.test_dir, img) for img in os.listdir(args.test_dir)]

    results = []

    for query_img_path in tqdm(query_img_paths, desc="Processing queries"):
        # Get embedding for query
        query_feature = get_query_embedding(model, query_img_path, device=device)
        
        # Find similar images using precomputed features
        similar_images = find_similar_images_with_precomputed(
            query_feature, dataset_features, dataset_paths, top_k=args.top_k
        )

        # Extract just the filenames for similar images
        similar_filenames = [os.path.basename(path) for path, _ in similar_images]
        
        results.append({
            "filename": os.path.basename(query_img_path),
            "samples": similar_filenames
        })

        # Display results if needed
        # print(f"Similar images for: {query_img_path}")
        # plot_query_and_similars(query_img_path, similar_images)
    
    # Write to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
