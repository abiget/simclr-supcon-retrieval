import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from get_images import (
    get_image_loader,
    extract_embeddings,
    get_top_k_similar,
    save_results_to_json,
    submit
)

# images paths
GALLERY_FOLDER = "data/test/gallery/"
QUERY_FOLDER = "data/test/query/"
K = 10  # number of most similar images to retrieve

# Load pretrained ResNet50 as a feature extractor (no classifier)


def load_pretrained_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_pretrained_model()
    model.to(device)

    # Load data
    print("📂 Loading gallery and query images...")
    gallery_loader = get_image_loader(GALLERY_FOLDER)
    query_loader = get_image_loader(QUERY_FOLDER)

    # Extract embeddings
    print("📸 Extracting embeddings...")
    gallery_embeddings = extract_embeddings(gallery_loader, model, device)
    print(f"✅ Extracted {len(gallery_embeddings)} gallery embeddings.")
    query_embeddings = extract_embeddings(query_loader, model, device)
    print(f"✅ Extracted {len(query_embeddings)} query embeddings.")

    # Compute similarities and save
    print(f"🔍 Retrieving top-{K} similar images...")
    results = get_top_k_similar(query_embeddings, gallery_embeddings, k=K)

    print("💾 Saving results...")
    save_results_to_json(results)

    submit(results, groupname="Beasts",
           url="http://65.108.245.177:3001/retrieval/")
