for i in {0..4};do
python train_v7.py --gpu_id 0 --nmute 0 --epochs 75 --nlayers 5 \
--batch_size 128 --lr_scale 0.07 --path data --workers 8 \
--dropout 0.1 --nclass 5 --ntoken 21 --nhead 16 --ninp 640 --nhid 2560 --warmup_steps 600 \
--fold $i --weight_decay 0.1 --nfolds 5 --error_alpha 0.5 --noise_filter 0.25
done
