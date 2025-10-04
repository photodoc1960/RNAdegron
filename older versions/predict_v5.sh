python predict_v5.py --gpu_id 0 --kmer_aggregation --batch_size 32 --kmers 1 \
--path ./data/kaggle --weights_path best_pl_weights --nfolds 5 \
--nclass 5 --ntoken 21 --nhead 8 --ninp 256 --nhid 1024 --dropout 0.1 --nlayers 5
