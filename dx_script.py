import torch
import os


def diagnose_checkpoint(checkpoint_path):
    """Comprehensive checkpoint diagnostic analysis."""
    print(f"\n🔍 Analyzing: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("❌ File does not exist")
        return False

    try:
        # Test basic loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Basic loading successful")
        print(f"📊 Checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"📋 Keys: {list(checkpoint.keys())}")

            # Check for state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"📦 State dict found with {len(state_dict)} parameters")
            else:
                state_dict = checkpoint
                print(f"📦 Direct state dict with {len(state_dict)} parameters")

            # Sample parameter analysis
            sample_keys = list(state_dict.keys())[:5]
            print(f"🔑 Sample parameter keys: {sample_keys}")

            # Check for DataParallel prefix
            has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
            print(f"🔗 DataParallel wrapper detected: {has_module_prefix}")

            return True

    except Exception as e:
        print(f"❌ Loading failed: {e}")
        return False


# Test multiple checkpoints
test_paths = [
    "./best_weights/fold0top1.ckpt",
    "./best_weights/fold0top2.ckpt",
    "./checkpoints_fold0/best_model.ckpt"
]

for path in test_paths:
    diagnose_checkpoint(path)