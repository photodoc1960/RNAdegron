#!/bin/bash
# CRITICAL FIX: V7 Pseudo-label training script with architectural compatibility

echo "🚀 Starting pseudo-label training with v7 architecture..."
echo "=================================================================="

# Validate pseudo-label availability
echo "📋 Validating pseudo-label files..."
missing_files=0
for fold in {0..4}; do
    file="../pseudo_labels/pseudo_labels_fold${fold}.p"
    if [ ! -f "$file" ]; then
        echo "❌ Missing pseudo-labels for fold ${fold}: $file"
        missing_files=$((missing_files + 1))
    else
        echo "✅ Found pseudo-labels for fold ${fold}"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "❌ Error: $missing_files pseudo-label files missing"
    echo "   Run: bash pseudo_predict_v7.sh first"
    exit 1
fi

echo ""
echo "🎯 Starting pseudo-label training for all folds..."

# CRITICAL FIX: Updated training loop with v7 compatibility
for i in {0..4}; do
    echo ""
    echo "=================================================================="
    echo "🔄 Training fold $i with pseudo-labels..."
    echo "=================================================================="
    
    # CRITICAL FIX: Updated script name and parameters for v7 architecture
    python train_pl_v7.py \
        --gpu_id 0 \
        --nmute 0 \
        --epochs 150 \
        --train_epochs 5 \
        --pl_epochs 2 \
        --rollback_thresh 0.002 \
        --nlayers 5 \
        --batch_size 64 \
        --lr_scale 0.07 \
        --path data \
        --workers 8 \
        --dropout 0.2 \
        --nclass 5 \
        --ntoken 21 \
        --nhead 16 \
        --ninp 640 \
        --nhid 2560 \
        --warmup_steps 600 \
        --fold $i \
        --weight_decay 0.1 \
        --nfolds 5 \
        --error_alpha 0.5 \
        --noise_filter 0.25 \
        --lr 1e-4 \
        --std_threshold 0.1 \
        --std_eps 1e-6

    # Check training success
    if [ $? -eq 0 ]; then
        echo "✅ Fold $i pseudo-label training completed successfully"
        
        # Validate output files
        checkpoint_dir="weights/checkpoints_fold${i}_pl"
        if [ -d "$checkpoint_dir" ]; then
            echo "📁 Generated checkpoints in: $checkpoint_dir"
            ls -la "$checkpoint_dir"/*.ckpt 2>/dev/null | head -3
        fi
        
        log_file="logs/log_pl_fold${i}.csv"
        if [ -f "$log_file" ]; then
            echo "📊 Training log: $log_file"
            echo "   Last few lines:"
            tail -3 "$log_file"
        fi
    else
        echo "❌ Fold $i pseudo-label training failed"
        echo "   Check logs for error details"
        exit 1
    fi
done

echo ""
echo "=================================================================="
echo "🎉 All pseudo-label training completed successfully!"
echo "=================================================================="

# Generate summary report
echo ""
echo "📈 Training Summary Report:"
echo "=========================="

total_checkpoints=0
for i in {0..4}; do
    checkpoint_dir="weights/checkpoints_fold${i}_pl"
    if [ -d "$checkpoint_dir" ]; then
        count=$(ls "$checkpoint_dir"/*.ckpt 2>/dev/null | wc -l)
        total_checkpoints=$((total_checkpoints + count))
        echo "  Fold $i: $count checkpoints"
        
        # Check for best model
        if [ -f "$checkpoint_dir/best_model.ckpt" ]; then
            echo "    ✅ Best model available"
        else
            echo "    ⚠️ Best model missing"
        fi
    else
        echo "  Fold $i: ❌ No checkpoint directory"
    fi
done

echo ""
echo "📊 Total checkpoints generated: $total_checkpoints"

# Validate log files
echo ""
echo "📋 Training Logs Summary:"
echo "========================"
for i in {0..4}; do
    log_file="logs/log_pl_fold${i}.csv"
    if [ -f "$log_file" ]; then
        lines=$(wc -l < "$log_file")
        echo "  Fold $i: $lines lines in log"
        
        # Extract final validation loss if available
        if [ $lines -gt 1 ]; then
            final_loss=$(tail -1 "$log_file" | cut -d',' -f3)
            if [ ! -z "$final_loss" ] && [ "$final_loss" != "val_loss" ]; then
                echo "    Final validation loss: $final_loss"
            fi
        fi
    else
        echo "  Fold $i: ❌ No log file"
    fi
done

echo ""
echo "🎯 Next Steps:"
echo "============="
echo "  1. Collect best weights: python get_best_weights_v7.py"
echo "  2. Run final predictions: bash predict_v7.sh"
echo "  3. Generate submission file"

echo ""
echo "✅ Pseudo-label training pipeline complete!"