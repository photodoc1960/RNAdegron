#!/bin/bash
# CRITICAL FIX: V7 Pseudo-label generation script with architectural compatibility

# Create output directory for pseudo-labels
mkdir -p ../pseudo_labels

echo "🚀 Starting pseudo-label generation with v7 architecture..."
echo "=================================================================="

# CRITICAL FIX: Updated parameters for v7 architecture compatibility
python pseudo_predict_v7.py \
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
  --output_dir ../pseudo_labels

# Validate pseudo-label generation success
if [ $? -eq 0 ]; then
    echo "✅ Pseudo-label generation completed successfully"
    echo "📊 Generated files:"
    ls -la ../pseudo_labels/pseudo_labels_fold*.p

    # Validate file sizes and content
    echo ""
    echo "📈 File validation:"
    for fold in {0..4}; do
        file="../pseudo_labels/pseudo_labels_fold${fold}.p"
        if [ -f "$file" ]; then
            size=$(stat --format="%s" "$file")
            echo "  Fold ${fold}: ${size} bytes"

            # Basic content validation using Python
            # Basic content validation using Python
            python -c "
import pickle
try:
    with open('${file}', 'rb') as f:
        data = pickle.load(f)
    print('    ✅ Valid pickle file with keys:', list(data.keys()) if isinstance(data, dict) else 'legacy format')
except Exception as e:
    print('    ❌ Error reading file:', str(e))
            "
        else
            echo "  ❌ Fold ${fold}: File not found"
        fi
    done
else
    echo "❌ Pseudo-label generation failed"
    exit 1
fi

echo ""
echo "🎯 Next steps:"
echo "  1. Run pseudo-label training: bash run_pl_v7.sh"
echo "  2. Combine with supervised predictions for final ensemble"

echo "=================================================================="
echo "✅ Pseudo-label generation pipeline complete"