# RNAdegron - RNA Degradation Prediction System

**Advanced deep learning pipeline for predicting RNA sequence degradation using RiNALMo embeddings**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## Table of Contents

- [About](#about)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [Recent Improvements](#recent-improvements)
- [Attribution](#attribution)
- [Citation](#citation)
- [License](#license)

---

## About

**RNAdegron** is an advanced deep learning system for predicting RNA degradation at individual nucleotide positions. Built upon the powerful [RiNALMo](https://github.com/lbcb-sci/RiNALMo) RNA language model, RNAdegron incorporates structural features and sophisticated training strategies to achieve state-of-the-art performance in RNA stability prediction.

### What RNAdegron Does

- **Predicts degradation likelihood** for each position in an RNA sequence
- **Incorporates structural information** including base pairing probabilities and graph distances
- **Uses ensemble predictions** across 5-fold cross-validation
- **Implements pseudo-labeling** for semi-supervised learning
- **Provides uncertainty quantification** for predictions

### Built Upon RiNALMo

This project leverages embeddings from **RiNALMo** (RiboNucleic Acid Language Model), a 650M parameter transformer pre-trained on 36 million RNA sequences. RNAdegron extends RiNALMo's capabilities specifically for degradation prediction tasks.

**Original RiNALMo:** [Paper](https://arxiv.org/abs/2403.00043) | [Code](https://github.com/lbcb-sci/RiNALMo) | [Weights](https://zenodo.org/records/15043668)

---

## Key Features

### 🧬 Advanced Architecture (v7)
- **Multi-modal inputs:** RiNALMo embeddings (640-dim) + structural features
- **Base pairing probabilities (BPP)** from ViennaRNA
- **Graph distance features** capturing RNA topology
- **Nearest paired/unpaired distances** for local structure
- **ΔG energy values** from thermodynamic calculations

### 🎯 Sophisticated Training
- **5-fold cross-validation** for robust model selection
- **Pseudo-labeling pipeline** for semi-supervised learning
- **Automatic rollback mechanism** to prevent degradation
- **Position-aware validation** with uncertainty filtering
- **Cluster-based sample weighting** for better generalization

### 🔧 Production-Ready
- **Multi-process safe data loading** with file locking
- **Memory-efficient** tensor operations
- **Robust error handling** with NaN detection
- **Comprehensive logging** and checkpointing
- **One-command pipeline execution**

---

## Repository Structure

```
RNAdegron/
├── README.md                      # This file
├── QUICK_START.md                # 5-minute getting started
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # Version history
├── LICENSE                       # Apache 2.0 license
├── .gitignore                    # Git ignore rules
│
├── environment.yml               # Conda environment
├── pyproject.toml               # Package configuration
├── full_pipeline_script.sh      # Complete pipeline automation
│
├── Core Pipeline (v7 - Current)
│   ├── Dataset_v7.py            # Data loading with structural features
│   ├── Functions_v7.py          # Training utilities
│   ├── X_Network_v7.py          # Transformer architecture
│   ├── train_v7.py              # Main training script
│   ├── train_pl_v7.py           # Pseudo-label training
│   ├── predict_v7.py            # Final prediction
│   ├── pseudo_predict_v7.py     # Pseudo-label generation
│   ├── pretrain_v7.py           # Optional pretraining
│   ├── serialize_embeddings_v7.py # Feature extraction
│   └── get_best_weights_v7.py   # Model selection
│
├── Utilities
│   ├── Logger.py                # CSV logging
│   ├── Metrics.py               # Evaluation metrics
│   ├── LrScheduler.py           # Learning rate scheduling
│   ├── cluster_weighting.py     # Cluster-based weighting
│   ├── position_aware_validation.py # Position-aware stopping
│   └── visualization_v7.py      # Result visualization
│
└── RiNALMo Integration
    └── rinalmo/                 # Original RiNALMo package (for embeddings)
        ├── data/                # Data utilities
        ├── model/               # Model architecture
        └── pretrained.py        # Pre-trained model loading
```

---

## Installation

### Prerequisites
- Python >= 3.8
- CUDA >= 11.8 (for GPU support)
- 16GB+ RAM recommended
- 50GB+ disk space for data and models

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/photodoc1960/RNAdegron.git
cd RNAdegron

# Create conda environment
conda env create -f environment.yml
conda activate rnadegron

# Install RiNALMo for embeddings
pip install git+https://github.com/lbcb-sci/RiNALMo.git
pip install flash-attn==2.3.2
```

### Option 2: Pip Install

```bash
git clone https://github.com/photodoc1960/RNAdegron.git
cd RNAdegron

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/lbcb-sci/RiNALMo.git
pip install flash-attn==2.3.2
```

### Download RiNALMo Weights

```bash
mkdir -p weights
cd weights

# Download RiNALMo pre-trained model (required for embeddings)
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt  # 650M params

cd ..
```

---

## Quick Start

### 1. Prepare Your Data

Place your RNA sequences in JSON format:
```json
{"id": "seq1", "sequence": "ACGUACGUACGU", "seq_length": 12}
{"id": "seq2", "sequence": "GCGCGCGCGCGC", "seq_length": 12}
```

### 2. Extract Features

```bash
python serialize_embeddings_v7.py \
    --data_path ./data \
    --output_path ./data/precomputed_features.pt \
    --model_name giga-v1
```

### 3. Run Complete Pipeline

```bash
# One command for the entire pipeline
bash full_pipeline_script.sh
```

This will:
1. Extract RiNALMo embeddings and structural features
2. Train models with 5-fold cross-validation
3. Generate pseudo-labels on unlabeled data
4. Fine-tune with pseudo-labels
5. Select best models
6. Generate final predictions

### 4. Get Predictions

Your results will be in `predictions_v7/submission_v7.csv`

---

## Usage Guide

### Complete Pipeline Execution

```bash
bash full_pipeline_script.sh
```

### Step-by-Step Execution

#### **Step 1: Feature Extraction**
```bash
python serialize_embeddings_v7.py \
    --data_path ./data \
    --output_path ./data/precomputed_features.pt \
    --model_name giga-v1
```

#### **Step 2: Training (5-fold CV)**
```bash
# Train all folds
bash run_v7.sh

# Or train specific fold
python train_v7.py \
    --path ./data \
    --fold 0 \
    --epochs 75 \
    --batch_size 24 \
    --ninp 640 \
    --nhead 16 \
    --nhid 2560 \
    --nlayers 5
```

#### **Step 3: Pseudo-Label Generation**
```bash
bash pseudo_predict_v7.sh

# Or for specific fold
python pseudo_predict_v7.py \
    --path ./data \
    --fold 0 \
    --weights_path ./weights
```

#### **Step 4: Pseudo-Label Training**
```bash
bash run_pl_v7.sh

# Or for specific fold
python train_pl_v7.py \
    --path ./data \
    --fold 0 \
    --epochs 150 \
    --train_epochs 5 \
    --pl_epochs 2 \
    --rollback_thresh 0.002
```

#### **Step 5: Model Selection**
```bash
python get_best_weights_v7.py \
    --fold 0 \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints_fold0
```

#### **Step 6: Final Prediction**
```bash
python predict_v7.py \
    --path ./data \
    --weights_path ./best_weights \
    --output_dir ./predictions_v7 \
    --nfolds 5
```

---

## Model Architecture

### Input Features

1. **RiNALMo Embeddings (640-dim)**
   - Pre-trained contextual representations
   - Extracted from giga-v1 model
   - Captures sequence patterns and motifs

2. **Structural Features**
   - **BPP Matrices:** Base pairing probabilities from ViennaRNA
   - **Graph Distances:** Topological distances in RNA structure
   - **Nearest Paired Distances:** Distance to closest paired base
   - **Nearest Unpaired Distances:** Distance to closest unpaired base
   - **ΔG Values:** Free energy of predicted structures

### Network Architecture

```
Input Layer (640-dim RiNALMo embeddings)
    ↓
Structural Feature Integration
    ↓
Multi-Head Attention with BPP (16 heads)
    ↓
Convolutional Transformer Encoder (5 layers, 2560 hidden)
    ↓
Output Layer (5 degradation targets)
```

### Training Strategy

- **Cross-Validation:** 5-fold stratified splits
- **Pseudo-Labeling:** Semi-supervised learning on unlabeled data
- **Alternating Training:** 5 epochs supervised + 2 epochs pseudo-labeled
- **Rollback Protection:** Automatic reversion if validation degrades
- **Position Weighting:** Focus on reliable regions
- **Cluster Weighting:** Balance sequence diversity

---

## Recent Improvements

### v7.1 Bug Fixes
- ✅ **Fixed sequence length assumptions** - Uses actual lengths from data
- ✅ **Implemented file locking** - Safe multi-process data loading
- ✅ **Added NaN handling** - Robust training with automatic rollback
- ✅ **Fixed memory leaks** - Proper tensor detachment
- ✅ **Initialized all variables** - No more NameError crashes

### Performance Enhancements
- ✅ **Multi-process safety** - Concurrent BPP generation with locks
- ✅ **Position-aware early stopping** - Better convergence detection
- ✅ **Winner's strategy integration** - Column weights, cluster sampling

---

## Attribution

### Built Upon RiNALMo

This project uses embeddings from **RiNALMo** (RiboNucleic Acid Language Model):

**RiNALMo Citation:**
```bibtex
@article{penic2024rinalmo,
  title={RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks},
  author={Peni{\'c}, Rafael Josip and Vla{\v{s}}i{\'c}, Tin and Huber, Roland G and Wan, Yue and {\v{S}}iki{\'c}, Mile},
  journal={arXiv preprint arXiv:2403.00043},
  year={2024}
}
```

**RiNALMo Resources:**
- 📄 [Paper](https://arxiv.org/abs/2403.00043)
- 💻 [Code](https://github.com/lbcb-sci/RiNALMo)
- 🔗 [Pre-trained Weights](https://zenodo.org/records/15043668)

### Additional Tools

- **ViennaRNA** for structure prediction and BPP calculation
- **PyTorch** for deep learning framework
- **Flash Attention** for efficient attention mechanisms

---

## Citation

If you use RNAdegron in your research, please cite:

```bibtex
@software{rnadegron2024,
  title={RNAdegron: RNA Degradation Prediction using RiNALMo Embeddings},
  author={photodoc1960},
  year={2024},
  url={https://github.com/photodoc1960/RNAdegron}
}
```

**Please also cite the original RiNALMo paper** (see [Attribution](#attribution) section).

---

## License

Copyright 2024 photodoc1960

### RNAdegron Code License
This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

### Third-Party Dependencies
This project uses RiNALMo for embedding extraction. RiNALMo is licensed under:
- **Code:** Apache License 2.0
- **Model Parameters:** CC BY 4.0

**Users must comply with RiNALMo's licensing terms** when using this software. Please cite both RNAdegron and RiNALMo in your work.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Support

- 📧 **Issues:** [GitHub Issues](https://github.com/photodoc1960/RNAdegron/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/photodoc1960/RNAdegron/discussions)
- 📖 **Documentation:** [Wiki](https://github.com/photodoc1960/RNAdegron/wiki) (coming soon)

---

## Acknowledgments

- **RiNALMo Team** at University of Zagreb and A*STAR for the foundational language model
- **ViennaRNA Team** for RNA structure prediction tools
- **PyTorch Community** for the deep learning framework
- All contributors to this project

---

**RNAdegron** - Advancing RNA degradation prediction through deep learning 🧬
