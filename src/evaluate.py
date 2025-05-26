import torch
import argparse
from torchvision import transforms
from tqdm import tqdm
from retrieve import get_query_embedding, find_similar_images_with_precomputed, precompute_dataset_embeddings
from utils.data_utils import ImageDataset
from utils.load_model_utils import load_model
from retrieve import evaluation_transform

def top_k_accuracy(model, data_dir, k=5, device='cpu', transform=None, mean=[0.5]*3, std=[0.5]*3):

    if transform is None:
        transform = evaluation_transform(img_size=32, mean=mean, std=std)

    dataset = ImageDataset(root=data_dir, transform=transform)

    if not hasattr(dataset, 'samples'):
        raise ValueError("Dataset must have 'samples' attribute with image paths and labels")
    
    if not dataset.use_folders:
        raise ValueError("The dataset must have class folders for top-k accuracy evaluation")

    # Precompute all dataset embeddings
    all_features, all_paths = precompute_dataset_embeddings(model, dataset, device)
    
    # Get mapping from path to label
    path_to_label = {img_path: label for img_path, label in dataset.samples}
    
    correct = 0
    total = 0

    # Track per-class accuracy
    class_correct = {cls: 0 for cls in dataset.classes}
    class_total = {cls: 0 for cls in dataset.classes}

    for query_path, query_label in tqdm(dataset.samples, desc=f"Evaluating Top-{k} Accuracy"):
        # Get embedding for query image
        query_feature = get_query_embedding(model, query_path, device)
        
        # Find similar images
        similar_images = find_similar_images_with_precomputed(
            query_feature, all_features, all_paths, top_k=k + 1  # +1 to account for the query image itself
        )
        
        # Remove the query image itself if it's in the results
        similar_images = [(path, score) for path, score in similar_images if path != query_path][:k]
        
        # Check if any of the top-k results have the same label as the query
        retrieved_labels = [path_to_label[path] for path, _ in similar_images]
        if query_label in retrieved_labels:
            correct += 1
            class_name = dataset.classes[query_label]
            class_total[class_name] += 1
        
        total += 1

    
    accuracy = correct / total
    print(f"Top-{k} accuracy: {accuracy:.4f}")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for cls in dataset.classes:
        cls_acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        print(f"{cls}: {cls_acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
    
    return accuracy
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Top-K Accuracy")
    parser.add_argument('--data_dir', default="data/test", type=str, help='Path to the dataset directory')
    parser.add_argument('--model_path', default="checkpoints/supcon_experiment/supcon_model_final.pth", type=str, required=True, help='Path to the trained model')
    parser.add_argument('--k', type=int, default=5, help='Top K for accuracy evaluation')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the evaluation on')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path, device=args.device, backbone_only=True)
    model.eval() # for safe measure, ensure model is in eval mode
    top_k_accuracy(model, args.data_dir, k=args.k, device=args.device)