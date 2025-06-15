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
   - Measure top-k retrieval score (9.55 to 781 on competition data)
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

We experimented with three different approaches:

1. **ResNet-50 with SimCLR Pretraining**:
   - Backbone: ResNet-50 pretrained using SimCLR on ImageNet
   - No fine-tuning, used as baseline
   - Best suited for: Natural image datasets similar to ImageNet domain

2. **FaceNet Pretrained**:
   - Used pretrained FaceNet model
   - Direct evaluation on competition dataset without fine-tuning
   - Specialized for face recognition tasks
   - Better suited for: Face-specific image retrieval

3. **SimCLR Fine-tuned with SupCon**:
   - ResNet-50 backbone with SimCLR pretraining
   - Fine-tuned using Supervised Contrastive Loss
   - Projection Head: MLP with structure [2048 → 512 → 128]
   - Optimized for: Better feature learning on target dataset

This architecture comparison revealed how model performance is heavily influenced by the alignment between pretraining domain and target dataset characteristics.

## Results and Performance

Our experimental workflow consisted of three main phases:

1. **Initial Evaluation on Competition Data**:
   - Evaluated pretrained SimCLR (ResNet-50 backbone) - achieved 9.55 score
   - Evaluated pretrained FaceNet model - achieved 781 score
   - FaceNet showed significantly better performance without any fine-tuning

2. **Competition Data Fine-tuning Analysis**:
   - Fine-tuned SimCLR model using SupCon loss on competition data
   - After 100 epochs, achieved 18 score
   - Training dynamics analysis revealed instability:
     - Initial phase: Loss decreased from around 3.7 to ~2.45 by epoch 20
     - Significant instability around epochs 21-27:
       - Sharp spike starting at epoch 21 (about 3.86)
       - Peaked at epoch 22 (4.59)
       - Gradual recovery to ~3.7 by epoch 26
     - This unstable behavior suggests fundamental issues:
       - Extreme class imbalance (some classes having only 1 example)
       - Insufficient positive pairs for contrastive learning
       - Domain mismatch between SimCLR's natural image pretraining and face recognition task
   - Results suggested this approach was unsuiPr 2table for such a small, imbalanced face dataset

3. **Intel Image Dataset Validation**:
   - To verify our approach wasn't fundamentally flawed, we switched to the larger Intel Image Classification dataset
   - Initial evaluation with pretrained SimCLR showed strong 68.8% accuracy, likely due to:
     - Intel dataset's similarity to ImageNet's general-purpose nature
     - Better alignment with SimCLR's pretraining on natural images
     - Sufficient examples per class for effective contrastive learning
   - After fine-tuning with SupCon loss, achieved excellent 97.77% accuracy
   - This dramatic improvement confirmed our hypothesis that the poor performance on the competition dataset was due to:
     - Extreme data scarcity
     - Domain mismatch (faces vs natural images)
     - Insufficient positive pairs for contrastive learning

### Key Findings

1. **Model-Data Compatibility**: 
   - FaceNet's superior performance (781 vs SimCLR's 9.55) demonstrates the importance of using domain-specific pretrained models for limited face data
   - SimCLR+SupCon performs exceptionally well (97.77%) when domain and data requirements are met

2. **Data Requirements for SupCon**: Training dynamics revealed that Supervised Contrastive Learning requires:
   - Multiple examples per class to create positive pairs
   - Reasonably balanced class distribution
   - Domain alignment with the pretraining dataset

3. **Solution Validation**: While the approach struggled with the face dataset, its success on the Intel dataset (97.77%) validates that SimCLR+SupCon is highly effective when these requirements are met

### Model Performance Comparison

The first three results on the competition data is based on the evaluation metrics given on the competition (the evaluation server results).

| Model Configuration | Dataset | Accuracy (%) | Notes |
|-------------------|----------|-------------|--------|
| FaceNet (pretrained, no fine-tuning) | Competition Data | 781 | Best performance on competition data |
| ResNet-50 (SimCLR pretrained) | Competition Data | 9.55 | Initial baseline, domain mismatch |
| ResNet-50 (SimCLR + 50 epochs SupCon) | Competition Data | 18 | Limited by extremely scarce data (1 image/class) |
| ResNet-50 (SimCLR pretrained) | Intel Image | Top-3: 68.8 | Strong baseline due to dataset similarity with ImageNet |
| ResNet-50 (SimCLR + SupCon fine-tuned) | Intel Image | Top-3: 97.77 | Excellent performance with sufficient data |

Note: The performance differences can be attributed to two key factors:

1. Dataset Domain: SimCLR's pretrained weights performed better on Intel dataset (68.8%) versus competition data (9.55 (though its score)) due to its similarity with ImageNet-style natural images
2. Training Data Availability: The competition dataset's extreme scarcity (some example contain 1 image per class) severely limited fine-tuning potential, while the Intel dataset provided sufficient examples for effective learning

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

#### 1. SimCLR Pretrained ResNet-50

```bash
python src/retrieve.py --model_type simclr \
    --query_dir data/competition/test/query \
    --gallery_dir data/competition/test/gallery
```

```bash
python src/retrieve.py --model_type simclr \
    --query_dir data/intel_dataset/test/query \
    --gallery_dir data/intel_dataset/test/gallery \
    --top_k 3 \
    --custom_dataset
```

#### 2. FaceNet Pretrained

```bash
python src/retrieve.py --model_type facenet \
    --query_dir data/competition/test/query \
    --gallery_dir data/competition/test/gallery
```

#### 3. SimCLR Fine-tuned with SupCon

```bash
python src/retrieve.py --model_type supcon-tuned \
    --model_path checkpoints/supcon_experiment_final/supcon_model_final.pth \
    --query_dir data/competition/test/query \
    --gallery_dir data/competition/test/gallery
```

### Fine-tuning

#### 1. Fine-tune SimCLR with SupCon on Competition Data

```bash
python src/fine_tune.py --model simclr \
    --train_dir data/train --epochs 50 --batch_size 128 \
    --save_dir checkpoints/supcon_experiment
```

#### 2. Fine-tune on Intel Image Dataset

```bash
python src/fine_tune.py --model simclr \
    --train_dir data/intel_dataset/train --epochs 100 --batch_size 128 \
    --save_dir checkpoints/supcon_experiment_intel_data
```

## Credits

This implementation builds upon the following excellent repositories:

1. [SupContrast](https://github.com/HobbitLong/SupContrast) - Implementation of Supervised Contrastive Learning
2. [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - PyTorch implementation of FaceNet used for baseline comparison
3. [lightly-ai/simclr](https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x) - Pretrained SimCLR ResNet-50 weights for feature extraction

## References

1. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
2. [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)
3. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

## License

[MIT License](LICENSE)
