import numpy as np
import os
import pandas as pd
import torch
from pathlib import Path
import shutil
from typing import List, Tuple, Optional
import warnings

def validate_checkpoint_integrity(checkpoint_path: str) -> bool:
    """
    Validate checkpoint file integrity and compatibility.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Boolean indicating if checkpoint is valid and loadable
    """
    try:
        if not os.path.exists(checkpoint_path):
            return False
            
        # Check file size (should be substantial for a trained model)
        file_size = os.path.getsize(checkpoint_path)
        if file_size < 1024 * 1024:  # Less than 1MB suggests corruption
            print(f"⚠️ Warning: Checkpoint {checkpoint_path} unusually small ({file_size} bytes)")
            return False
            
        # Attempt to load checkpoint header to verify format
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if not isinstance(state_dict, dict) or len(state_dict) == 0:
                print(f"❌ Invalid checkpoint format: {checkpoint_path}")
                return False
            print(f"✅ Checkpoint validated: {checkpoint_path} ({file_size / (1024*1024):.1f}MB)")
            return True
        except Exception as e:
            print(f"❌ Checkpoint load error: {checkpoint_path} - {e}")
            return False
            
    except Exception as e:
        print(f"❌ Checkpoint validation failed: {checkpoint_path} - {e}")
        return False

def parse_training_log_robust(csv_file: str) -> pd.DataFrame:
    """
    Parse training log with enhanced error handling and format validation.
    
    Args:
        csv_file: Path to CSV training log
        
    Returns:
        Pandas DataFrame with validated training metrics
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Training log not found: {csv_file}")
    
    try:
        # Load CSV with error handling
        history = pd.read_csv(csv_file)
        
        # Validate required columns
        required_columns = ['epoch', 'val_loss']
        missing_columns = [col for col in required_columns if col not in history.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {csv_file}: {missing_columns}")
        
        # Clean and validate data
        initial_rows = len(history)
        
        # Filter out invalid entries
        valid_mask = (
            history['val_loss'].notna() & 
            (history['val_loss'] != -1) & 
            (history['val_loss'] != 0) &
            (history['val_loss'] < 100) &  # Reasonable upper bound
            (history['epoch'].notna()) &
            (history['epoch'] > 0)
        )
        
        history_clean = history[valid_mask].copy()
        filtered_rows = initial_rows - len(history_clean)
        
        if len(history_clean) == 0:
            raise ValueError(f"No valid validation scores found in {csv_file}")
        
        if filtered_rows > 0:
            print(f"📊 Filtered {filtered_rows} invalid entries from {csv_file}")
        
        print(f"📈 Loaded {len(history_clean)} valid training epochs from {csv_file}")
        print(f"   Validation loss range: {history_clean['val_loss'].min():.6f} - {history_clean['val_loss'].max():.6f}")
        
        return history_clean
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse training log {csv_file}: {e}")

def find_available_checkpoints(weights_path: str, epochs: List[int]) -> List[Tuple[int, str]]:
    """
    Find available checkpoint files for specified epochs with fallback mechanisms.
    
    Args:
        weights_path: Directory containing checkpoint files
        epochs: List of epoch numbers to search for
        
    Returns:
        List of (epoch, checkpoint_path) tuples for available checkpoints
    """
    available_checkpoints = []
    
    if not os.path.exists(weights_path):
        print(f"❌ Weights directory not found: {weights_path}")
        return available_checkpoints
    
    for epoch in epochs:
        # Primary checkpoint naming convention
        checkpoint_patterns = [
            f"epoch{epoch}.ckpt",
            f"epoch{int(epoch)}.ckpt",
            f"model_epoch_{epoch}.ckpt",
            f"checkpoint_epoch_{epoch}.pt",
            "best_model.ckpt"  # Fallback to best model if specific epoch not found
        ]
        
        found = False
        for pattern in checkpoint_patterns:
            checkpoint_path = os.path.join(weights_path, pattern)
            if os.path.exists(checkpoint_path) and validate_checkpoint_integrity(checkpoint_path):
                available_checkpoints.append((epoch, checkpoint_path))
                found = True
                break
        
        if not found:
            print(f"⚠️ No valid checkpoint found for epoch {epoch} in {weights_path}")
    
    return available_checkpoints

def get_best_weights_from_fold_v7(fold: int, 
                                  csv_file: str, 
                                  weights_path: str, 
                                  destination: str, 
                                  top: int = 5) -> Tuple[float, List[float]]:
    """
    Extract best performing model weights from a training fold with comprehensive validation.
    
    Args:
        fold: Fold number (0-4)
        csv_file: Path to training log CSV
        weights_path: Directory containing checkpoint files
        destination: Output directory for best weights
        top: Number of top models to extract
        
    Returns:
        Tuple of (best_score, all_top_scores)
    """
    print(f"\n🔍 Processing fold {fold}...")
    print(f"   Log file: {csv_file}")
    print(f"   Weights path: {weights_path}")
    
    try:
        # Parse training log with validation
        history = parse_training_log_robust(csv_file)
        
        # Convert validation loss to scores (negative for maximization)
        scores = -history['val_loss'].values
        epochs = history['epoch'].values
        
        # Find top performing epochs
        top_indices = scores.argsort()[-top:][::-1]
        top_scores = scores[top_indices]
        top_epochs = epochs[top_indices]
        
        print(f"📊 Top {len(top_scores)} validation scores: {top_scores}")
        print(f"🎯 Corresponding epochs: {top_epochs}")
        
        # Find available checkpoints
        available_checkpoints = find_available_checkpoints(weights_path, top_epochs)
        
        if not available_checkpoints:
            raise RuntimeError(f"No valid checkpoints found for fold {fold}")
        
        # Create destination directory
        os.makedirs(destination, exist_ok=True)
        
        # Copy best available checkpoints
        copied_count = 0
        final_scores = []
        
        for i, (epoch, checkpoint_path) in enumerate(available_checkpoints):
            if i >= top:  # Limit to requested number
                break
                
            destination_path = os.path.join(destination, f"fold{fold}top{i+1}.ckpt")
            
            try:
                # Copy checkpoint with verification
                shutil.copy2(checkpoint_path, destination_path)
                
                # Verify copied file
                if validate_checkpoint_integrity(destination_path):
                    # Find corresponding score
                    epoch_mask = history['epoch'] == epoch
                    if epoch_mask.any():
                        score = -history.loc[epoch_mask, 'val_loss'].iloc[0]
                        final_scores.append(score)
                        print(f"✅ Copied fold{fold}top{i+1}.ckpt (epoch {epoch}, score: {score:.6f})")
                        copied_count += 1
                    else:
                        print(f"⚠️ Could not find score for epoch {epoch}")
                else:
                    print(f"❌ Copied checkpoint failed validation: {destination_path}")
                    if os.path.exists(destination_path):
                        os.remove(destination_path)
                        
            except Exception as e:
                print(f"❌ Failed to copy checkpoint for epoch {epoch}: {e}")
        
        if copied_count == 0:
            raise RuntimeError(f"No checkpoints successfully copied for fold {fold}")
        
        best_score = max(final_scores) if final_scores else top_scores[0]
        print(f"🏆 Fold {fold} best score: {best_score:.6f} ({copied_count} models copied)")
        
        return best_score, final_scores
        
    except Exception as e:
        print(f"❌ Error processing fold {fold}: {e}")
        raise

def validate_cross_validation_results(scores: List[float], 
                                     fold_details: List[List[float]]) -> dict:
    """
    Perform comprehensive statistical analysis of cross-validation results.
    
    Args:
        scores: List of best scores from each fold
        fold_details: List of all top scores for each fold
        
    Returns:
        Dictionary with statistical analysis results
    """
    scores_array = np.array(scores)
    
    # Basic statistics
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    # Cross-validation confidence interval (95%)
    confidence_interval = 1.96 * std_score / np.sqrt(len(scores))
    
    # Performance consistency analysis
    coefficient_variation = std_score / mean_score if mean_score != 0 else float('inf')
    
    # Fold-wise analysis
    fold_analysis = []
    for i, fold_scores in enumerate(fold_details):
        if fold_scores:
            fold_mean = np.mean(fold_scores)
            fold_std = np.std(fold_scores) if len(fold_scores) > 1 else 0
            fold_analysis.append({
                'fold': i,
                'best_score': max(fold_scores),
                'mean_score': fold_mean,
                'std_score': fold_std,
                'num_models': len(fold_scores)
            })
    
    return {
        'mean_cv_score': mean_score,
        'std_cv_score': std_score,
        'min_score': min_score,
        'max_score': max_score,
        'confidence_interval': confidence_interval,
        'coefficient_variation': coefficient_variation,
        'fold_analysis': fold_analysis,
        'num_folds': len(scores)
    }

def generate_performance_report(cv_results: dict, output_path: str = 'model_selection_report.txt'):
    """Generate comprehensive performance report."""
    
    report_lines = [
        "=" * 80,
        "MODEL SELECTION PERFORMANCE REPORT - V7 ARCHITECTURE",
        "=" * 80,
        "",
        "CROSS-VALIDATION SUMMARY:",
        f"  Mean CV Score: {cv_results['mean_cv_score']:.6f}",
        f"  Standard Deviation: {cv_results['std_cv_score']:.6f}",
        f"  95% Confidence Interval: ±{cv_results['confidence_interval']:.6f}",
        f"  Score Range: {cv_results['min_score']:.6f} - {cv_results['max_score']:.6f}",
        f"  Coefficient of Variation: {cv_results['coefficient_variation']:.4f}",
        f"  Number of Folds: {cv_results['num_folds']}",
        "",
        "FOLD-WISE ANALYSIS:",
        "-" * 40
    ]
    
    for fold_data in cv_results['fold_analysis']:
        report_lines.extend([
            f"Fold {fold_data['fold']}:",
            f"  Best Score: {fold_data['best_score']:.6f}",
            f"  Mean Score: {fold_data['mean_score']:.6f}",
            f"  Std Score: {fold_data['std_score']:.6f}",
            f"  Models Available: {fold_data['num_models']}",
            ""
        ])
    
    # Performance assessment
    if cv_results['coefficient_variation'] < 0.05:
        stability = "EXCELLENT (Very Stable)"
    elif cv_results['coefficient_variation'] < 0.1:
        stability = "GOOD (Stable)"
    elif cv_results['coefficient_variation'] < 0.2:
        stability = "MODERATE (Some Variation)"
    else:
        stability = "POOR (High Variation)"
    
    report_lines.extend([
        "PERFORMANCE ASSESSMENT:",
        f"  Model Stability: {stability}",
        f"  Recommendation: {'Ensemble ready' if cv_results['coefficient_variation'] < 0.15 else 'Consider additional training'}",
        "",
        "=" * 80
    ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MODEL SELECTION SUMMARY")
    print("=" * 60)
    print(f"🎯 Mean CV Score: {cv_results['mean_cv_score']:.6f} ± {cv_results['std_cv_score']:.6f}")
    print(f"📈 Performance Stability: {stability}")
    print(f"📁 Detailed report saved: {output_path}")

def main():
    """Main execution function for best weights collection."""
    
    print("🚀 Starting V7 Best Weights Collection...")
    print("=" * 60)
    
    # Configuration
    num_folds = 5
    top_models_per_fold = 5
    base_destination = 'best_weights'
    
    # Collect best weights from all folds
    fold_scores = []
    fold_details = []
    successful_folds = 0
    
    for fold in range(num_folds):
        try:
            # CRITICAL FIX: Support both supervised and pseudo-label training
            csv_paths = [
                f"log_fold{fold}.csv",  # Supervised training
                f"logs/log_pl_fold{fold}.csv",  # Pseudo-label training
                f"logs/log_fold{fold}.csv"  # Alternative location
            ]

            weights_paths = [
                f"checkpoints_fold{fold}",  # Supervised training
                f"weights/checkpoints_fold{fold}_pl",  # Pseudo-label training
                f"checkpoints_fold{fold}_pl"  # Alternative naming
            ]

            # Find existing paths
            csv_file = next((path for path in csv_paths if os.path.exists(path)), csv_paths[0])
            weights_path = next((path for path in weights_paths if os.path.exists(path)), weights_paths[0])

            print(f"📁 Using CSV: {csv_file}")
            print(f"📁 Using weights: {weights_path}")
            
            best_score, all_scores = get_best_weights_from_fold_v7(
                fold=fold,
                csv_file=csv_file,
                weights_path=weights_path,
                destination=base_destination,
                top=top_models_per_fold
            )
            
            fold_scores.append(best_score)
            fold_details.append(all_scores)
            successful_folds += 1
            
        except Exception as e:
            print(f"❌ Failed to process fold {fold}: {e}")
            fold_scores.append(float('-inf'))  # Placeholder for failed fold
            fold_details.append([])
    
    # Validate results
    if successful_folds == 0:
        raise RuntimeError("No folds processed successfully")
    
    # Filter out failed folds
    valid_scores = [score for score in fold_scores if score != float('-inf')]
    valid_details = [details for details in fold_details if details]
    
    if len(valid_scores) < num_folds:
        print(f"⚠️ Warning: Only {len(valid_scores)}/{num_folds} folds processed successfully")
    
    # Perform statistical analysis
    cv_results = validate_cross_validation_results(valid_scores, valid_details)
    
    # Generate comprehensive report
    generate_performance_report(cv_results)
    
    # Save cross-validation score for external use
    final_cv_score = cv_results['mean_cv_score']
    with open('cv.txt', 'w') as f:
        f.write(f"{-final_cv_score:.8f}")  # Negative for loss representation
    
    print(f"\n✅ Cross-validation score saved: {-final_cv_score:.8f}")
    print(f"📁 Best weights collected in: {base_destination}/")
    
    # List final collected weights
    print(f"\n📋 Collected Weights Summary:")
    if os.path.exists(base_destination):
        weight_files = sorted([f for f in os.listdir(base_destination) if f.endswith('.ckpt')])
        for i, weight_file in enumerate(weight_files, 1):
            file_path = os.path.join(base_destination, weight_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {i:2d}. {weight_file} ({file_size:.1f}MB)")
    
    print("\n🎉 Best weights collection completed successfully!")

if __name__ == '__main__':
    main()