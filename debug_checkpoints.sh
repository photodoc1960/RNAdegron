#!/bin/bash

echo "🔍 Checking checkpoint directories..."

for fold in {0..4}; do
    echo ""
    echo "Fold $fold:"
    echo "  Expected: weights/checkpoints_fold${fold}_pl/epoch59.ckpt or epoch60.ckpt"
    
    if [ -d "weights/checkpoints_fold${fold}_pl" ]; then
        echo "  📁 Directory exists"
        echo "  📊 Files found:"
        ls -la "weights/checkpoints_fold${fold}_pl/" | head -10
        echo "  📊 Total files: $(ls weights/checkpoints_fold${fold}_pl/ | wc -l)"
        
        # Check for epoch 59 and 60 specifically
        if [ -f "weights/checkpoints_fold${fold}_pl/epoch59.ckpt" ]; then
            echo "  ✅ epoch59.ckpt EXISTS"
        else
            echo "  ❌ epoch59.ckpt MISSING"
        fi
        
        if [ -f "weights/checkpoints_fold${fold}_pl/epoch60.ckpt" ]; then
            echo "  ✅ epoch60.ckpt EXISTS"
        else
            echo "  ❌ epoch60.ckpt MISSING"
        fi
    else
        echo "  ❌ Directory doesn't exist"
    fi
done

echo ""
echo "🔍 Checking for any existing checkpoints..."
find . -name "*.ckpt" -type f 2>/dev/null | head -20
