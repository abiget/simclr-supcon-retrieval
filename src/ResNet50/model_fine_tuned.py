'''In this section we use transfer learning with a pretrained ResNet50 model. 
Transfer learning is a machine learning technique where a pretrained model is taken (usually trained on a big task) 
and then reused for a different but related task. 
This technique is really convenient because 
training a deep learning model from scratch requires a lot of data 
and takes a long time and lots of computing power. 
With transfer learning, however, we can save resources and get better performance 
even with small datasets. [explain better in the report and cite]
'''

import torch
import torch.nn as nn
from torchvision.models import resnet50
from get_images import (
    get_image_loader,
    extract_embeddings,
    get_top_k_similar,
    save_results_to_json,
    submit
)

'''general steps:
1. Use transfer learning with a pretrained ResNet50 model.
2. Replace its final classification layer to match your classes.
3. Fine-tune the model on your training images.
4. Save the model.
5. Remove the classification head so you can use it to extract features for your image retrieval pipeline.
'''

# File paths
GALLERY_FOLDER = "data/test/gallery/"
QUERY_FOLDER = "data/test/query/"
FINE_TUNED_PATH = "fine_tuned_model.pth"
K = 10

# Load fine-tuned ResNet50 as feature extractor


def load_finetuned_model(path):
    print(f"🧠 Loading fine-tuned model from: {path}")

    # Step 1: Create full model matching the saved one
    model = resnet50(weights=None)

    # Step 2: Temporarily match the fc layer to what was saved
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Get the correct number of output classes from the checkpoint
    num_classes = state_dict["fc.weight"].shape[0]
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Step 3: Load weights (fc now matches, so no error)
    model.load_state_dict(state_dict, strict=False)

    # Step 4: Remove classifier to use as feature extractor
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fine-tuned model
    model = load_finetuned_model(FINE_TUNED_PATH)
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
