# RNAdegron Quick Start Guide

Get started with RNA degradation prediction in 5 minutes! 🚀

## What is RNAdegron?

**RNAdegron** predicts RNA degradation at individual nucleotide positions using deep learning. It combines:
- RiNALMo embeddings (from the original RiNALMo language model)
- RNA structural features
- Advanced training strategies

## 1. Installation (2 minutes)

### Option A: Conda (Recommended)
```bash
git clone https://github.com/photodoc1960/RNAdegron
cd RNAdegron
conda env create -f environment.yml
conda activate rnadegron
pip install git+https://github.com/lbcb-sci/RiNALMo.git
pip install flash-attn==2.3.2
```

### Option B: Pip
```bash
git clone https://github.com/photodoc1960/RNAdegron
cd RNAdegron
pip install -r requirements.txt
pip install git+https://github.com/lbcb-sci/RiNALMo.git
pip install flash-attn==2.3.2
```

## 2. Get Pre-trained Weights (1 minute)

```bash
mkdir weights && cd weights
# Download RiNALMo weights (required for embeddings)
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt
cd ..
```

## 3. Prepare Your Data

Create a JSON file with RNA sequences:
```json
{"id": "seq1", "sequence": "ACGUACGUACGU", "seq_length": 12}
{"id": "seq2", "sequence": "GGCCGGCCGGCC", "seq_length": 12}
```

## 4. Run Prediction Pipeline (2 minutes)

### Option 1: One Command (Full Pipeline)
```bash
bash full_pipeline_script.sh
```

### Option 2: Step by Step

```bash
# Extract features
python serialize_embeddings_v7.py --data_path ./data

# Train models (5-fold CV)
bash run_v7.sh

# Generate predictions
bash predict_v7.sh
```

## 5. Get Results

Your predictions will be in: `predictions_v7/submission_v7.csv`

---

## Common Use Cases

### Predict degradation for your sequences

```bash
# 1. Put sequences in data/your_sequences.json
# 2. Extract features
python serialize_embeddings_v7.py \
    --data_path ./data \
    --output_path ./data/precomputed_features.pt

# 3. Run prediction with pre-trained models
python predict_v7.py \
    --path ./data \
    --weights_path ./best_weights \
    --output_dir ./my_predictions
```

### Train on your own degradation data

```bash
# 1. Prepare training data in JSON format
# 2. Extract features
python serialize_embeddings_v7.py --data_path ./data

# 3. Train
python train_v7.py \
    --path ./data \
    --fold 0 \
    --epochs 75 \
    --batch_size 24
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_v7.py --batch_size 12  # instead of 24
```

### Missing Dependencies
```bash
# Reinstall
conda env remove -n rnadegron
conda env create -f environment.yml
conda activate rnadegron
```

### Can't find RiNALMo weights
```bash
# Make sure weights are downloaded
ls weights/rinalmo_giga_pretrained.pt
# If missing, re-download
cd weights && wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt
```

---

## Pipeline Overview

```
Your RNA Sequences
        ↓
Extract RiNALMo Embeddings + Structural Features
        ↓
Train Models (5-fold CV)
        ↓
Generate Pseudo-Labels
        ↓
Fine-tune with Pseudo-Labels
        ↓
Select Best Models
        ↓
Final Predictions
```

---

## Key Features

✅ **Position-level predictions** - Degradation at each nucleotide
✅ **Structural awareness** - Uses BPP and graph distances
✅ **Ensemble predictions** - 5-fold cross-validation
✅ **Uncertainty quantification** - Know prediction confidence
✅ **Semi-supervised learning** - Pseudo-labeling for unlabeled data

---

## What's Included

| Component | Purpose |
|-----------|---------|
| `serialize_embeddings_v7.py` | Extract RiNALMo features |
| `train_v7.py` | Main training |
| `train_pl_v7.py` | Pseudo-label training |
| `predict_v7.py` | Generate predictions |
| `full_pipeline_script.sh` | Run everything |

---

## Next Steps

- 📖 Read the full [README](README.md) for detailed documentation
- 🔬 Explore [model architecture](README.md#model-architecture)
- 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- 📝 Check [CHANGELOG.md](CHANGELOG.md) for updates

---

## Attribution

RNAdegron uses **RiNALMo** for embeddings:
- 📄 [RiNALMo Paper](https://arxiv.org/abs/2403.00043)
- 💻 [RiNALMo Code](https://github.com/lbcb-sci/RiNALMo)

**Please cite both RNAdegron and RiNALMo in your work!**

---

**Need Help?**
- 📧 [Open an issue](https://github.com/photodoc1960/RNAdegron/issues)
- 💬 [Start a discussion](https://github.com/photodoc1960/RNAdegron/discussions)

Happy RNA degradation prediction! 🧬
