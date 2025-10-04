#!/usr/bin/env python3
"""
Verification script to ensure structural features are properly set up
"""

import os
import torch
import numpy as np
from X_Network_v6 import RNADegformer

def check_structural_features():
    """Check if structural features file exists and is valid."""
    print("🔍 Checking structural features...")
    
    feat_path = 'data/precomputed_features.pt'
    if not os.path.exists(feat_path):
        print(f"❌ Structural features file not found: {feat_path}")
        print("   Run: python serialize_embeddings_v6.py")
        return False
    
    try:
        feat_data = torch.load(feat_path, weights_only=False)
        required_keys = ['ids', 'embeddings', 'graph_dists', 'nearest_paired', 'nearest_unpaired']
        
        for key in required_keys:
            if key not in feat_data:
                print(f"❌ Missing key in features file: {key}")
                return False
        
        num_sequences = len(feat_data['ids'])
        print(f"✅ Structural features loaded: {num_sequences} sequences")
        print(f"   - Embeddings shape: {feat_data['embeddings'].shape}")
        print(f"   - Graph distances shape: {feat_data['graph_dists'].shape}")
        print(f"   - Nearest paired shape: {feat_data['nearest_paired'].shape}")
        print(f"   - Nearest unpaired shape: {feat_data['nearest_unpaired'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading structural features: {e}")
        return False

def check_model_architecture():
    """Check if model can be created with structural features."""
    print("\n🔍 Checking model architecture...")
    
    try:
        model = RNADegformer(
            ntoken=21, 
            nclass=5, 
            ninp=256, 
            nhead=8, 
            nhid=1024,
            nlayers=5, 
            kmer_aggregation=True, 
            kmers=[1], 
            stride=1,
            dropout=0.1, 
            pretrain=False,
            rinalmo_weights_path='/home/slater/RiNALMo/weights/rinalmo_micro_pretrained.pt'
        )
        
        # Check if structural feature layers exist
        has_graph_proj = hasattr(model, 'graph_proj')
        has_seq_proj = hasattr(model, 'seq_proj')
        has_struct_proj = hasattr(model, 'struct_proj')
        has_dg_proj = hasattr(model, 'dg_proj')
        has_joint_decoder = hasattr(model, 'joint_decoder')
        
        print(f"✅ Model created successfully")
        print(f"   - Graph projection layer: {'✅' if has_graph_proj else '❌'}")
        print(f"   - Sequence projection layer: {'✅' if has_seq_proj else '❌'}")
        print(f"   - Structural projection layer: {'✅' if has_struct_proj else '❌'}")
        print(f"   - ΔG projection layer: {'✅' if has_dg_proj else '❌'}")
        print(f"   - Joint decoder: {'✅' if has_joint_decoder else '❌'}")
        
        if all([has_graph_proj, has_seq_proj, has_struct_proj, has_dg_proj, has_joint_decoder]):
            print("✅ All structural feature layers present")
            return True
        else:
            print("❌ Some structural feature layers missing")
            return False
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def check_training_data():
    """Check if training data exists."""
    print("\n🔍 Checking training data...")
    
    train_path = 'data/train.json'
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return False
    
    test_path = 'data/test.json'
    if not os.path.exists(test_path):
        print(f"❌ Test data not found: {test_path}")
        return False
    
    print("✅ Training and test data found")
    return True

def main():
    print("🧪 RNA Degradation Prediction Setup Verification")
    print("=" * 50)
    
    checks = [
        check_training_data(),
        check_structural_features(),
        check_model_architecture()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 All checks passed! Ready for training with structural features.")
        print("\nTo start training:")
        print("1. bash run_v6.sh          # Supervised training")
        print("2. bash pseudo_predict_v6.sh # Generate pseudo-labels")  
        print("3. bash run_pl_v6.sh        # Pseudo-label training")
        print("4. python fixed_get_best_weights.py # Select best weights")
        print("5. bash predict_v6.sh       # Final predictions")
        print("\nOr run the full pipeline:")
        print("bash full_pipeline.sh")
    else:
        print("❌ Some checks failed. Please fix the issues above before training.")
        
        if not checks[1]:  # Structural features failed
            print("\n💡 To fix structural features:")
            print("   python serialize_embeddings_v6.py")

if __name__ == '__main__':
    main()
