for i in {0..4};do
python train_v3.py --gpu_id 0 --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 128 --kmers 1 --lr_scale 0.1 --path data --workers 6 \
--dropout 0.1 --nclass 5 --ntoken 21 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
--fold $i --weight_decay 0.1 --nfolds 5 --error_alpha 0.5 --noise_filter 0.25 \
--rinalmo_weights_path "/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt"
done
