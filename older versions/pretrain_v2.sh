for i in {0..0};do
python pretrain_v2.py --gpu_id 0 --nmute 0 --epochs 200 --nlayers 5 \
--batch_size 24 --kmers 1 --lr_scale 0.1 --path data --workers 2 \
--dropout 0.1 --nclass 5 --ntoken 21 --nhead 8 --ninp 256 --nhid 1024 --warmup_steps 600 \
--fold $i --weight_decay 0.1
done

