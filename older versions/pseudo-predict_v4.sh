#!/bin/bash
# Script to generate pseudo-labels for semi-supervised training

# Create output directory
mkdir -p ../pseudo_labels

# Run pseudo-label generation with the same parameters as training scripts
python pseudo_predict_v4.py \
  --gpu_id 0 \
  --kmer_aggregation \
  --batch_size 16 \
  --kmers 5 \
  --path ./data \
  --weights_path best_weights \
  --nfolds 10 \
  --nclass 5 \
  --ntoken 21 \
  --nhead 8 \
  --ninp 256 \
  --nhid 1024 \
  --dropout 0.1 \
  --nlayers 5 \
  --output_dir ../pseudo_labels

echo "Pseudo-label generation complete"
