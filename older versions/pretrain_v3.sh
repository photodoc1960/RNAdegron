#!/bin/bash

set -e

# Create save folders if missing
mkdir -p pretrain_weights
mkdir -p logs
mkdir -p visualizations_best_model

# Main loop (currently only fold 0)
for i in {0..0}; do
python pretrain_v3.py \
--gpu_id 0 \
--nmute 0 \
--epochs 200 \
--nlayers 5 \
--batch_size 128 \
--kmers 1 \
--lr_scale 0.1 \
--path data \
--workers 2 \
--dropout 0.1 \
--nclass 5 \
--ntoken 21 \
--nhead 8 \
--ninp 256 \
--nhid 1024 \
--warmup_steps 600 \
--fold $i \
--weight_decay 0.1
done

# (Optional) Backup logs and visualizations
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p runs/$timestamp
if [ -d logs ]; then
    cp -r logs runs/$timestamp/
fi

if [ -d visualizations_best_model ]; then
    cp -r visualizations_best_model runs/$timestamp/
fi

echo "Training complete. Best model remains at pretrain_weights/best_model.ckpt"
