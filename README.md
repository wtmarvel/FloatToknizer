# Float Value Tokenizer

A PyTorch implementation of VAE/VQ-VAE for float value tokenization. This project provides a distributed training framework with various features like mixed precision training, EMA, and flexible configuration management.

## Features
- Distributed training support (DDP)
- Mixed precision training (AMP)
- Multiple model architectures:
  - VAE (Variational Autoencoder)
  - VQ-VAE (Vector Quantized VAE)
- Exponential Moving Average (EMA)
- Flexible configuration system (supports .py, .yaml, .json)
- TensorBoard logging
- Automatic checkpoint management

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU(s)

### Setup
```bash
# Clone the repository
git clone git@github.com:wtmarvel/FloatToknizer.git
cd FloatToknizer

# Install dependencies
pip install -r requirements.txt
```

## Training

### Quick Start
There are several ways to start training. Choose the one that best suits your needs:

1. Multi-GPU Training (Recommended)
```bash
torchrun --nproc_per_node=2 main_multi_nodes.py \
    --config=configs/config_debug.py \
    --batch_size_per_gpu=512
```

2. Single GPU Training
```bash
python main_multi_nodes.py --config=configs/config_debug.py
```

### Configuration

The project supports multiple configuration file formats:
- Python (.py)
- YAML (.yaml, .yml)
- JSON (.json)

Key configuration parameters:
```python
# Model Structure
num_embeddings = 512    # Size of codebook (VQ-VAE only)
embedding_dim = 16      # Dimension of latent space
num_tokens = 4         # Number of tokens per float value (VQ-VAE only)

# Training
batch_size_per_gpu = 512
learning_rate = 1e-4
total_steps = 1_000_000
enable_amp = True      # Mixed precision training

# Optimization
gradient_accumulation_steps = 1
warm_steps = 1000
weight_decay = 0.1
```

You can override any configuration parameter via command line:
```bash
torchrun --nproc_per_node=2 main_multi_nodes.py \
    --config=configs/config_debug.py \
    --batch_size_per_gpu=64 \
    --