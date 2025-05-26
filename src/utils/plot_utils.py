import matplotlib.pyplot as plt
from PIL import Image

def plot_image(image_path, title=None, ax=None):
    """Plot a single image"""
    img = Image.open(image_path).convert('RGB')
    if ax is None:
        plt.imshow(img)
        plt.axis('off')
        if title is not None:
            plt.title(title, fontsize=10)
    else:
        ax.imshow(img)
        ax.axis('off')
        if title is not None:
            ax.set_title(title, fontsize=10)

def plot_query_and_similars(query_path, similar_images):
    """
    Plots the query image and its most similar images.
    similar_images: list of (image_path, similarity_score)
    """
    n_similars = len(similar_images)
    fig, axes = plt.subplots(1, n_similars + 1, figsize=(3*(n_similars+1), 3))
    
    # Plot query image
    plot_image(query_path, title='Query', ax=axes[0])
    
    # Plot similar images
    for i, (img_path, score) in enumerate(similar_images):
        title = f"Sim: {score:.3f}"
        plot_image(img_path, title=title, ax=axes[i+1])
    
    plt.tight_layout()
    plt.show()