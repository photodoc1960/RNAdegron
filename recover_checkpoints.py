import os
import shutil
import glob


def recover_checkpoints():
    print("🔧 Recovering existing checkpoints...")
    os.makedirs('best_pl_weights', exist_ok=True)

    checkpoint_dirs = glob.glob('checkpoints_fold*')

    for checkpoint_dir in checkpoint_dirs:
        try:
            fold_num = int(checkpoint_dir.split('_fold')[1])
            print(f"Processing fold {fold_num} from {checkpoint_dir}")

            checkpoint_files = glob.glob(f"{checkpoint_dir}/epoch*.ckpt")
            if checkpoint_files:
                epoch_numbers = []
                for file in checkpoint_files:
                    epoch_str = file.split('epoch')[1].split('.ckpt')[0]
                    try:
                        epoch_numbers.append((int(epoch_str), file))
                    except ValueError:
                        continue

                if epoch_numbers:
                    latest_epoch, latest_file = max(epoch_numbers)
                    dest_file = f"best_pl_weights/fold{fold_num}top1.ckpt"
                    shutil.copy2(latest_file, dest_file)
                    print(f"  ✅ Copied {latest_file} -> {dest_file}")

        except (ValueError, IndexError) as e:
            print(f"  ❌ Error processing {checkpoint_dir}: {e}")

    print("🔧 Checkpoint recovery complete")


if __name__ == "__main__":
    recover_checkpoints()
