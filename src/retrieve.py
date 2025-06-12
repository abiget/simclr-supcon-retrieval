import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from utils.data_utils import get_data_loader
from utils.load_model_utils import load_model
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import load_data_statistics
from submit import submit
from fine_tune import download_simclr_checkpoint

def extract_backbone_features(model, images, device='cpu'):
    """
    Extract embeddings from the model's backbone.
    Args:
        model (nn.Module): The model from which to extract embeddings.
        images (torch.Tensor): Input images of shape [batch_size, channels, height, width].
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        np.ndarray: Extracted embeddings as a NumPy array.
    """
    model.eval()
    with torch.no_grad():
        # If model is SupConModel, get backbone
        if hasattr(model, "backbone"):
            features = model.backbone(images.to(device))
        else:
            features = model(images.to(device))
    return features.cpu().numpy()

def evaluation_transform(img_size=160, mean=[0.5]*3, std=[0.5]*3):
    """
    Create a transform for evaluation.
    Args:
        img_size (int): Size of the input images (assumed square).
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
    Returns:
        transforms.Compose: A composed transform for evaluation.
    """

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def normalize_feature(feature):
    return feature / np.linalg.norm(feature, axis=1, keepdims=True)

def precompute_dataset_embeddings(model, dataset, device='cpu', img_size=160):
    """
    Precompute embeddings for the entire gallery dataset.
    Args:
        model (nn.Module): The model to use for feature extraction.
        dataset (Dataset): The dataset containing images.
        device (str): Device to run the model on ('cpu' or 'cuda').
        img_size (int): Size of the input images (assumed square).
    """
    model.eval()
    model = model.to(device)

    transform = evaluation_transform(img_size=img_size)

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
    Find top K similar images using precomputed features.
    Args:
        query_feature (np.ndarray): Feature vector of the query image.
        all_features (np.ndarray): Precomputed features of the dataset.
        all_paths (list): List of paths corresponding to the precomputed features.
        top_k (int): Number of top similar images to retrieve.
    Returns:
        list: List of tuples containing (image_path, similarity_score) for top K similar images.
    """
    # Compute cosine similarity since normalized features are used
    similarities = np.dot(all_features, query_feature.T).flatten()

    # Get top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    similar_images = [(all_paths[i], similarities[i]) for i in top_indices]

    return similar_images

def get_query_embedding(model, image_path, device='cpu', img_size=160):
    """
    Get the embedding for a single query image.
    Args:
        model (nn.Module): The model to use for feature extraction.
        image_path (str): Path to the query image.
        device (str): Device to run the model on ('cpu' or 'cuda').
        img_size (int): Size of the input images (assumed square).
    Returns:
        np.ndarray: Normalized feature vector for the query image.
    """
    model.eval()
    model = model.to(device)

    transform = evaluation_transform(img_size=img_size)

    # Extract feature for the query image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    query_feature = extract_backbone_features(model, image_tensor, device)
    # Normalize query feature
    query_feature = normalize_feature(query_feature)
    
    return query_feature


if __name__ == "__main__":
    """Main function to run the image retrieval process."""
    parser = argparse.ArgumentParser(description="Image Retrieval using SupCon finetune Model or Facenet")
    parser.add_argument("--model_path", type=str, default="checkpoints/supcon_experiment_intel_data/supcon_model_final.pth", help="Path to the trained model")
    parser.add_argument("--gallery_dir", type=str, default="data/test/gallery/", help="Directory containing gallery images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader")
    parser.add_argument("--query_dir", type=str, default="data/test/query/", help="Directory containing query images for retrieval")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top similar images to retrieve")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--output_file", type=str, default="results/submission.json", help="Path to save the output JSON file")
    parser.add_argument("--image_size", type=int, default=160, help="Size of the input images (assumed square)")
    parser.add_argument("--stat_file", type=str, default="results/dataset_statistics.pth", help="Path to the dataset statistics file")
    parser.add_argument("--data_dir", type=str, default="data/train/", help="Directory containing training images")
    parser.add_argument("--model_type", type=str, default="facenet", choices=["simclr", "facenet", "supcon-tuned"], help="Type of model to load (simclr or facenet)")
    parser.add_argument("--custom_dataset", action='store_true', help="Flag to indicate if using a custom dataset for testing")
    parser.add_argument("--simclr_restnet_checkpoint", type=str, default="resnet50-1x.pth", help="Path to the SimCLR ResNet50 checkpoint if using SimCLR pretrained weights")
    args = parser.parse_args()
    # Load the model
    device = args.device

    if not args.model_type in ["facenet", "simclr"] and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Please check the path.")

    if args.model_type == "simclr" or args.model_type == "simclr-tuned":
        if not os.path.exists(args.simclr_restnet_checkpoint):
            # download it form haggingface
            download_simclr_checkpoint(pretrained_path=args.simclr_restnet_checkpoint)

    # Load the model, assuming it has a method to load the full model
    model = load_model(args.model_path, device=device, backbone_only=True, model_type=args.model_type)

    # compute or load dataset statistics
    mean, std = load_data_statistics(args.stat_file, data_dir=args.data_dir, image_size=args.image_size)

    # get the gallery data loader
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
    dataset_features, dataset_paths = precompute_dataset_embeddings(model, gallery_loader.dataset, device=device, img_size=args.image_size)
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
    submission_dict = {}

    # Iterate over each query image and find similar images
    for query_img_path, label in tqdm(query_dataset.samples, desc="Processing queries"):
        # Get embedding for query
        query_feature = get_query_embedding(model, query_img_path, device=device)
        
        # Find similar images usingis_train precomputed features
        similar_images = find_similar_images_with_precomputed(
            query_feature, dataset_features, dataset_paths, top_k=args.top_k
        )

        # Extract just the filenames for similar images
        similar_filenames = [os.path.basename(path) for path, _ in similar_images]
        
        # save results for custom dataset testing
        results.append({
            "filename": os.path.basename(query_img_path),
            "label": query_dataset.classes[label] if query_dataset.use_folders else label,
            "samples": [path for path, _ in similar_images]
        })

        # Prepare submission data
        submission_dict[os.path.basename(query_img_path)] = similar_filenames

    # Write to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if not args.custom_dataset:
        group_name = "Beasts"

        print(f"Submitting results for group: {group_name}")

        submit(submission_dict, group_name)
    else:
