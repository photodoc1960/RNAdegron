for i in {0..4}; do
    python train_pl_v6.py \
        --gpu_id 0 \
        --kmer_aggregation \
        --nmute 0 \
        --epochs 150 \
        --train_epochs 5 \
        --pl_epochs 2 \
        --rollback_thresh 0.002 \
        --nlayers 5 \
        --batch_size 128 \
        --kmers 1 \
        --lr_scale 0.1 \
        --path data \
        --workers 8 \
        --dropout 0.2 \
        --nclass 5 \
        --ntoken 21 \
        --nhead 8 \
        --ninp 256 \
        --nhid 1024 \
        --warmup_steps 600 \
        --fold $i \
        --weight_decay 0.1 \
        --nfolds 5 \
        --error_alpha 0.5 \
        --noise_filter 0.25
done
