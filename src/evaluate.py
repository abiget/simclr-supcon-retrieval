
import json
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# calculate top-k accuracy
def calculate_top_k_accuracy(path_to_submission, top_k=5):
    """
    Calculate the top-k accuracy from a submission file and return the overall accuracy and per-class accuracy.
    Args:
        path_to_submission (str): Path to the submission JSON file.
        top_k (int): The number of top predictions to consider for accuracy calculation.
    Returns:
        tuple: Overall accuracy and a dictionary with per-class accuracy."""

    with open(path_to_submission) as f:
        submission = json.load(f)
        
    if not isinstance(submission, list):
        raise ValueError("Submission must be a list")

    class_correct = {}
    class_total = {}
    correct = 0
    for item in submission:
        label = item['label']

        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0

        if label in [match.split('/')[-2] for match in item['samples'][:top_k]]:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1
    
    accuracy_per_class = {cls: (class_correct[cls] / class_total[cls]) * 100 for cls in class_correct}
    accuracy = correct / len(submission) * 100

    return accuracy, accuracy_per_class

def visualize_embeddings_with_tsne(features, labels, title="t-SNE Visualization of Embeddings"):
    """
    Visualize embeddings using t-SNE.

    Args:
        features: Feature embeddings from the model
        labels: Corresponding class labels
        title: Plot title
    """
    # Ensure inputs are numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    if len(features) != len(labels):
        raise ValueError(f"Number of features ({len(features)}) must match number of labels ({len(labels)})")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_tsne = tsne.fit_transform(features)
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    
    # Create plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], 
                    color=mcolors.TABLEAU_COLORS[colors[i % len(colors)]], 
                    label=str(label), alpha=0.6)
    
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{title.replace(" ", "_")}.png', dpi=300)
    plt.show()