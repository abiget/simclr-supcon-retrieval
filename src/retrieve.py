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
from utils.data_utils import compute_dataset_statistics, load_data_statistics
from submit import submit

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

    parser = argparse.ArgumentParser(description="Image Retrieval using SupCon Model")
    parser.add_argument("--model_path", type=str, default="checkpoints/supcon_experiment1/supcon_model_final.pth", help="Path to the trained model")
    parser.add_argument("--gallery_dir", type=str, default="data/test/gallery/", help="Directory containing gallery images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader")
    parser.add_argument("--query_dir", type=str, default="data/test/query/", help="Directory containing query images for retrieval")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top similar images to retrieve")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--output_file", type=str, default="submission.json", help="Path to save the output JSON file")
    parser.add_argument("--image_size", type=int, default=64, help="Size of the input images (assumed square)")
    parser.add_argument("--stat_file", type=str, default="dataset_statistics.pth", help="Path to the dataset statistics file")
    parser.add_argument("--data_dir", type=str, default="data/train/", help="Directory containing training images")
    args = parser.parse_args()
    # Load the model
    device = args.device

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Please check the path.")
    
    # Load the model, assuming it has a method to load the full model
    model = load_model(args.model_path, device=device, backbone_only=True)

    mean, std = load_data_statistics(args.stat_file, data_dir=args.data_dir, image_size=args.image_size)

    gallery_loader = get_data_loader(
        data_dir=args.gallery_dir,
        batch_size=args.batch_size,
        num_workers=4,
        is_train=False,
        image_size=args.image_size,
        mean=mean,
        std=std
    )

    # Precompute dataset embeddings
    dataset_features, dataset_paths = precompute_dataset_embeddings(model, gallery_loader.dataset, device=device)
    labels = gallery_loader.dataset.classes if gallery_loader.dataset.use_folders else None

    # # Process multiple query images efficiently
    query_dataset = get_data_loader(
        data_dir=args.query_dir,
        batch_size=1,  # Process one image at a time
        num_workers=4,
        is_train=False,
        image_size=args.image_size,
        mean=mean,
        std=std
    ).dataset

    results = []

    for query_img_path, label in tqdm(query_dataset.samples, desc="Processing queries"):
        # Get embedding for query
        query_feature = get_query_embedding(model, query_img_path, device=device)
        
        # Find similar images using precomputed features
        similar_images = find_similar_images_with_precomputed(
            query_feature, dataset_features, dataset_paths, top_k=args.top_k
        )

        # Extract just the filenames for similar images
        # similar_filenames = [os.path.basename(path) for path, _ in similar_images]
        similar_filenames = [path for path, _ in similar_images]
        
        results.append({
            "filename": os.path.basename(query_img_path),
            "label": query_dataset.classes[label] if query_dataset.use_folders else label,
            "samples": similar_filenames
        })

        # Display results if needed
        # print(f"Similar images for: {query_img_path}")
        # plot_query_and_similars(query_img_path, similar_images)
    
    # Write to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # load the json file into dict and submit the results
    with open(args.output_file, 'r') as f:
        submission_data = json.load(f)
    
    group_name = "beasts" 

    print(f"Submitting results for group: {group_name}")
    # print(submission_data)

    # submit(submission_data, group_name)
