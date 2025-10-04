# pretrain_multiarch.sh
mkdir -p pretrain_weights_depth{3,4,5,6,7}

for depth in {3..7}; do
  python pretrain.py --gpu_id 0 --kmer_aggregation --nmute 0 --epochs 200 --nlayers $depth \
  --batch_size 96 --kmers $depth --lr_scale 0.1 --path data --workers 2 \
  --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
  --fold 0 --weight_decay 0.1
done