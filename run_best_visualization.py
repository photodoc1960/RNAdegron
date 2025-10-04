from visualization import visualize_best_model

if __name__ == "__main__":
    visualize_best_model(
        model_path='pretrain_weights/best_model.ckpt',
        data_path='./data/kaggle/test.json',
        save_path='visualizations_best_model',
        rinalmo_path='./weights/rinalmo_micro_pretrained.pt',
        device='cuda:0',
        batch_size=1,
        mask_ratio=0.15,
        layer_idx=0,
        head_idx=0,
        seq_idx=0
    )