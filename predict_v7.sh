#!/bin/bash
# CRITICAL FIX: V7 Prediction script with architectural compatibility

echo "🚀 Starting V7 RNA degradation prediction pipeline..."
echo "=================================================================="

# Validate prerequisites
echo "📋 Validating prerequisites..."

# Check for required files
required_files=(
    "data/precomputed_features.pt"
    "data/test.json"
    "best_weights"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        echo "❌ Missing required file/directory: $file"
        missing_files=$((missing_files + 1))
    else
        echo "✅ Found: $file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "❌ Error: $missing_files required files missing"
    echo "   Ensure precomputed features and model weights are available"
    exit 1
fi

# Check for model weights
weight_count=$(find best_weights -name "fold*top*.ckpt" 2>/dev/null | wc -l)
if [ $weight_count -lt 5 ]; then
    echo "⚠️ Warning: Only $weight_count model weights found (expected 25 for 5 folds)"
    echo "   Continuing with available weights..."
else
    echo "✅ Found $weight_count model weight files"
fi

echo ""
echo "🎯 Executing V7 prediction with optimized parameters..."

# CRITICAL FIX: V7 architecture parameters aligned with training configuration
python predict_v7.py \
    --gpu_id 0 \
    --batch_size 8 \
    --path ./data \
    --weights_path best_weights \
    --nfolds 5 \
    --nclass 5 \
    --ntoken 21 \
    --nhead 16 \
    --ninp 640 \
    --nhid 2560 \
    --dropout 0.1 \
    --nlayers 5 \
    --output_dir predictions_v7 \
    --use_sliding_window \
    --window_size 130 \
    --stride 65

# Check execution success
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ V7 prediction completed successfully!"
    echo "=================================================================="
    
    # Validate output files
    echo "📊 Output validation:"
    output_dir="predictions_v7"
    
    if [ -d "$output_dir" ]; then
        echo "✅ Output directory created: $output_dir"
        
        # Check for detailed predictions
        if [ -f "$output_dir/predictions_v7_detailed.pkl" ]; then
            size=$(stat --format="%s" "$output_dir/predictions_v7_detailed.pkl")
            echo "✅ Detailed predictions: ${size} bytes"
        else
            echo "❌ Missing detailed predictions file"
        fi
        
        # Check for submission file
        if [ -f "$output_dir/submission_v7.csv" ]; then
            lines=$(wc -l < "$output_dir/submission_v7.csv")
            echo "✅ Submission file: ${lines} lines"
            
            # Show sample of submission file
            echo ""
            echo "📋 Submission file preview:"
            head -5 "$output_dir/submission_v7.csv"
        else
            echo "⚠️ No submission file generated (template may be missing)"
        fi
        
        # List all output files
        echo ""
        echo "📁 Generated files:"
        ls -la "$output_dir/"
    else
        echo "❌ Output directory not created"
    fi
    
    echo ""
    echo "🎯 Next steps:"
    echo "  1. Validate submission format"
    echo "  2. Check prediction quality metrics"
    echo "  3. Submit results"
    
else
    echo ""
    echo "❌ V7 prediction failed!"
    echo "=================================================================="
    echo "🔍 Troubleshooting suggestions:"
    echo "  1. Check GPU memory availability"
    echo "  2. Verify model weight integrity"
    echo "  3. Ensure precomputed features are compatible"
    echo "  4. Review error logs above"
    exit 1
fi

echo "=================================================================="
echo "✅ V7 prediction pipeline complete!"
