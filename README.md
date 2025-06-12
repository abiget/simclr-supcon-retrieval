# Image Retrieval with SimCLR and Supervised Contrastive Learning

## Project Overview

This project explores different approaches to image retrieval, comparing the effectiveness of ResNet-50 (pretrained with SimCLR) with FaceNet on both a competition dataset and the Intel Image Classification dataset. Through our experiments, we demonstrate how dataset domain similarity and data availability significantly impact model performance. While ResNet-50 (SimCLR) struggles with limited face image data, it excels on the Intel dataset due to better domain alignment with its ImageNet pretraining.

## Features

- Comparative analysis of different image retrieval approaches:
  - ResNet-50 pretrained with SimCLR (with and without fine-tuning)
  - FaceNet pretrained model evaluation
  - Performance comparison on both limited and large-scale datasets
- Implementation of Supervised Contrastive Learning for fine-tuning
- Comprehensive evaluation framework:
  - Top-k retrieval accuracy metrics
  - Per-class performance analysis
  - t-SNE visualization of embedding spaces
- Visual result analysis with query-retrieval pairs
- Support for both competition data and Intel Image Classification dataset

## How it Works

1. **Feature Learning Pipeline**:
   - Baseline evaluation with pretrained models (SimCLR ResNet50 and FaceNet)
   - Fine-tuning experiments with Supervised Contrastive Loss
   - Performance validation on larger Intel Image dataset

2. **Image Retrieval Process**:
   - Extract feature embeddings using the selected model
   - Compute cosine similarity between query and gallery images
   - Return the most similar images based on similarity scores
   - Evaluate retrieval accuracy using class labels

3. **Performance Analysis**:
   - Measure top-k retrieval accuracy (9.55% to 78.10% on competition data)
   - Demonstrate dramatic improvement with sufficient data (63% to 98.78% on Intel dataset)
   - Visualize embedding space organization using t-SNE
   - Analyze per-class performance and retrieval examples

## Requirements

- Python 3.7+
- ipykernel==6.29.5
- matplotlib==3.10.3
- torchvision==0.22.0+cu128
- tqdm==4.67.1
- wandb==0.19.11


## Model Architecture

We experimented with two different approaches, each with distinct advantages for different types of data:

1. **ResNet-50 with SimCLR**:
   - Backbone: ResNet-50 pretrained using SimCLR on ImageNet
   - Projection Head: MLP with structure [2048 → 512 → 128]
   - Training: Fine-tuned with Supervised Contrastive Loss
   - Best suited for: Natural image datasets similar to ImageNet domain

2. **FaceNet Model**:
   - Used pretrained FaceNet model
   - Direct evaluation on competition dataset without fine-tuning
   - Specialized for face recognition tasks
   - Better suited for: Face-specific image retrieval

This architecture comparison revealed how model performance is heavily influenced by the alignment between pretraining domain and target dataset characteristics.

## Results and Performance

Our experimental workflow consisted of three main phases:

1. **Initial Evaluation on Competition Data**:
   - Evaluated pretrained SimCLR (ResNet-50 backbone) - achieved 9.55% accuracy
   - Evaluated pretrained FaceNet model - achieved 78.10% accuracy
   - FaceNet showed significantly better performance without any fine-tuning

2. **Competition Data Fine-tuning Attempt**:
   - Fine-tuned SimCLR model using SupCon loss on competition data
   - After 50 epochs, achieved 18.42% accuracy
   - Limited improvement suggested issues with the small training dataset

3. **Intel Image Dataset Validation**:
   - To verify our approach, we switched to the larger Intel Image Classification dataset
   - Initial evaluation with pretrained SimCLR showed strong 63% accuracy, likely due to:
     - Intel dataset's similarity to ImageNet's general-purpose nature
     - Better alignment with SimCLR's pretraining on natural images
   - After fine-tuning with SupCon loss, achieved excellent 98.78% accuracy
   - This dramatic improvement validated our approach, confirming that performance was primarily limited by data availability in the competition dataset

### Model Performance Comparison

| Model Configuration | Dataset | Accuracy (%) | Notes |
|-------------------|----------|-------------|--------|
| FaceNet (pretrained, no fine-tuning) | Competition Data | 78.10 | Best performance on competition data |
| ResNet-50 (SimCLR pretrained) | Competition Data | 9.55 | Initial baseline, domain mismatch |
| ResNet-50 (SimCLR + 50 epochs SupCon) | Competition Data | 18.42 | Limited by extremely scarce data (1 image/class) |
| ResNet-50 (SimCLR pretrained) | Intel Image | 63.00 | Strong baseline due to dataset similarity with ImageNet |
| ResNet-50 (SimCLR + SupCon fine-tuned) | Intel Image | 98.78 | Excellent performance with sufficient data |

Note: The performance differences can be attributed to two key factors:

1. Dataset Domain: SimCLR's pretrained weights performed better on Intel dataset (63%) versus competition data (9.55%) due to its similarity with ImageNet-style natural images
2. Training Data Availability: The competition dataset's extreme scarcity (1 image per class) severely limited fine-tuning potential, while the Intel dataset provided sufficient examples for effective learning

### Visualization Results

#### Intel Image Dataset Retrieval Results

![Intel Dataset Retrieval Results](/results/similar_images_query_in_gallery_intel_dataset1.png)

#### Competition Dataset Retrieval Results

![Competition Data Retrieval Results](/results/similar_images_query_in_gallery_competition_data.png)

#### Embedding Space Visualizations

The t-SNE visualizations below demonstrate the embedding space organization. The Intel Image dataset results show clear clustering after fine-tuning with sufficient data, validating our approach despite the challenges with the competition dataset.

<!-- ##### FaceNet Embeddings on Competition Data

![t-SNE Visualization of FaceNet Embeddings](/results/t-SNE_Visualization_-_FaceNet_Embeddings.png) -->

##### ResNet-50 fine tuned with supcon Embeddings on Intel Dataset (100 epochs)

![t-SNE Visualization of Intel Dataset](/results/t-SNE_Visualization_Intel_Image_Dataset_Embeddings.png)

## Usage

### Evaluation

#### 1. ResNet-50 (SimCLR) Pretrained Evaluation

```bash
python src/retrieve.py --model simclr --weights resnet50-1x.pth \
    --query_dir data/competition/test/query \
    --gallery_dir data/competition/test/gallery
```

#### 2. FaceNet Evaluation

```bash
python src/retrieve.py --model facenet \
    --query_dir data/competition/test/query \
    --gallery_dir data/competition/test/gallery
```

#### 3. Fine-tuned SimCLR+SupCon Evaluation

```bash
python src/retrieve.py --model simclr_supcon \
    --weights checkpoints/supcon_experiment/supcon_model_final.pth \
    --query_dir data/competition/test/query \
    --gallery_dir data/competition/test/gallery
```

### Fine-tuning

#### 1. Fine-tune SimCLR with SupCon on Competition Data

```bash
python src/fine_tune.py --model simclr --pretrained resnet50-1x.pth \
    --train_dir data/train --epochs 50 --batch_size 32 \
    --save_dir checkpoints/supcon_experiment
```

#### 2. Fine-tune on Intel Image Dataset

```bash
python src/fine_tune.py --model simclr --pretrained resnet50-1x.pth \
    --train_dir data/intel_dataset/train --epochs 100 --batch_size 32 \
    --save_dir checkpoints/supcon_experiment_intel_data
```

## Credits

This implementation builds upon the following excellent repositories:

1. [SupContrast](https://github.com/HobbitLong/SupContrast) - Implementation of Supervised Contrastive Learning
2. [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - PyTorch implementation of FaceNet used for baseline comparison

## References

1. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
2. [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)
3. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

## License

[MIT License](LICENSE)
