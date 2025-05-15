# Multigeometry AI Framework

A sophisticated Python-based framework for training multimodal artificial intelligence models. This system is engineered to learn rich, shared representations between diverse types of data—such as images, text, audio, and source code—by processing them in pairs.

## Key Features

- **Multimodal Learning**: Supports training on diverse data modalities (e.g., images, text, audio, source code) by processing them in pairs.
- **Flexible Experiment Configuration**: Define dataset pairings, model architectures, and training parameters through dedicated data classes.
- **Scalable Data Pipelines**: Handles both map-style datasets and large-scale streaming datasets.
- **Comprehensive Preprocessing**: Includes pipelines for image augmentations and audio feature extraction using pre-trained speech models.
- **Probabilistic Representations**: Models each data sample as a mixture of Gaussians, capturing uncertainty and complex relationships.

## Core Components

### Configuration
- **`ModelConfig` and `DatasetPairConfig`**: Define and manage experimental setups and dataset parameters.

### Data Loading and Preprocessing
- **`PairDataset` and `IterableStylePairDataset`**: Efficiently handle map-style and streaming datasets.

### Model Architecture
- **Encoders**: Modality-specific encoders (e.g., `ImageEncoder`, `TextEncoder`) for feature extraction.
- **`PairModel`**: Processes paired inputs from different modalities.
- **`MixtureEmbedding`**: Converts encoder outputs into probabilistic representations (mixtures of Gaussians).

### Loss Functions
- **`MainCriterion`**: Computes a multi-component loss, including:
  - InfoNCE-like contrastive loss
  - Entropy regularization for mixture distributions
  - Optional `SinkhornOTLoss` for aligning distributions using optimal transport.

### Training Infrastructure
- Robust training loop with:
  - Checkpointing
  - Optimizer and learning rate scheduler setup
  - Gradient accumulation
  - Detailed logging
  - Validation loop for performance monitoring

## Mathematical Foundation

The framework represents each data sample \(x\) as a probability distribution, specifically a mixture of Gaussians:

\[
p(z|x) = \sum_k \gamma_k(x) \cdot \mathcal{N}(z | \mu_k(x), \Sigma_k(x))
\]

Where:
- \(\gamma_k(x)\): Mixture weights
- \(\mu_k(x)\): Mean vectors
- \(\Sigma_k(x)\): Covariance matrices (diagonal for simplicity)

The `MainCriterion` loss function aligns distributions \(p(z|x_i)\) and \(p(z|y_i)\) for corresponding
