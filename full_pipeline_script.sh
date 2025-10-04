#!/bin/bash

echo "🚀 Starting full RNA degradation prediction pipeline with structural features"
echo "=================================================================="

# Step 1: Ensure structural features are precomputed
echo "Step 1: Computing RiNALMo embeddings and structural features..."
python serialize_embeddings_v7.py
if [ $? -eq 0 ]; then
    echo "✅ Structural features computed successfully"
else
    echo "❌ Failed to compute structural features"
    exit 1
fi

# Step 2: Pretraining (optional - skip if you have pretrained weights)
echo ""
echo "Step 2: Pretraining (optional)..."
read -p "Do you want to run pretraining? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash pretrain_v7.sh
    if [ $? -eq 0 ]; then
        echo "✅ Pretraining completed successfully"
    else
        echo "❌ Pretraining failed"
        exit 1
    fi
else
    echo "⏭️  Skipping pretraining"
fi

# Step 3: Supervised training
echo ""
echo "Step 3: Supervised training with structural features..."
bash run_v7.sh
if [ $? -eq 0 ]; then
    echo "✅ Supervised training completed successfully"
else
    echo "❌ Supervised training failed"
    exit 1
fi

# Step 4: Generate pseudo-labels
echo ""
echo "Step 4: Generating pseudo-labels..."
bash pseudo_predict_v7.sh
if [ $? -eq 0 ]; then
    echo "✅ Pseudo-label generation completed successfully"
else
    echo "❌ Pseudo-label generation failed"
    exit 1
fi

# Step 5: Pseudo-label training
echo ""
echo "Step 5: Training with pseudo-labels..."
bash run_pl_v7.sh
if [ $? -eq 0 ]; then
    echo "✅ Pseudo-label training completed successfully"
else
    echo "❌ Pseudo-label training failed"
    exit 1
fi

# Step 6: Select best weights
echo ""
echo "Step 6: Selecting best model weights..."
python get_best_weights_v7.py
if [ $? -eq 0 ]; then
    echo "✅ Best weights selected successfully"
else
    echo "❌ Best weight selection failed"
    exit 1
fi

# Step 7: Final prediction
echo ""
echo "Step 7: Running final predictions..."
bash predict_v7.sh
if [ $? -eq 0 ]; then
    echo "✅ Predictions completed successfully"
else
    echo "❌ Prediction failed"
    exit 1
fi

# Step 8: FIXED - Comprehensive output validation
echo ""
echo "Step 8: Validating pipeline outputs..."

# Validate submission file exists and has content
if [ ! -f "predictions_v7/submission_v7.csv" ] || [ ! -s "predictions_v7/submission_v7.csv" ]; then
    echo "❌ Critical error: submission.csv missing or empty"
    exit 1
fi

# Validate submission content integrity
python3 -c "
import pandas as pd
import numpy as np
import sys

try:
    df = pd.read_csv('submission.csv')

    # Check basic structure
    if df.empty:
        print('❌ Submission file contains no data')
        sys.exit(1)

    # Check for NaN/infinite values
    if df.isnull().sum().sum() > 0:
        print('❌ Submission contains NaN values')
        sys.exit(1)

    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        print('❌ Submission contains infinite values')
        sys.exit(1)

    print(f'✅ Submission validated: {len(df)} predictions, {len(df.columns)} columns')

except Exception as e:
    print(f'❌ Submission validation failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Pipeline outputs validated successfully"
else
    echo "❌ Pipeline output validation failed - results may be corrupted"
    exit 1
fi

echo ""
echo "🎉 Complete pipeline finished successfully!"
echo "Check the output directory for submission files."
