# ADVANCED STRATEGY 3: Position-Aware Validation Framework
# Add this module as position_aware_validation.py

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings


class PositionAwareValidator:
    """
    Advanced validation framework acknowledging position-dependent prediction characteristics
    and pseudo-labeling corruption of traditional cross-validation.
    """

    def __init__(self, sequence_lengths, position_ranges=None, target_columns=None):
        """
        Initialize position-aware validation framework.

        Args:
            sequence_lengths: Array of original sequence lengths (107, 130)
            position_ranges: Dict defining position regions for analysis
            target_columns: Target prediction columns
        """
        self.sequence_lengths = sequence_lengths
        self.target_columns = target_columns or ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        # Winner's insight: positions >91 have fundamentally different characteristics
        self.position_ranges = position_ranges or {
            'reliable': (0, 68),  # High confidence predictions
            'moderate': (68, 91),  # Moderate confidence
            'uncertain': (91, 130)  # High variance region (excluded from scoring)
        }

        self.validation_history = defaultdict(list)
        self.position_statistics = {}

    def compute_position_variance_profile(self, predictions_ensemble, true_labels=None):
        """
        Analyze prediction variance across sequence positions.

        Args:
            predictions_ensemble: List of prediction arrays from different models
            true_labels: Ground truth labels (if available)

        Returns:
            Dictionary with position-wise variance statistics
        """
        if not predictions_ensemble:
            return {}

        # Stack predictions: (n_models, n_sequences, n_positions, n_targets)
        stacked_preds = np.stack(predictions_ensemble, axis=0)

        # Compute variance across model ensemble
        ensemble_variance = np.var(stacked_preds, axis=0)  # (n_sequences, n_positions, n_targets)

        # Aggregate statistics by position
        position_stats = {}
        n_positions = ensemble_variance.shape[1]

        for pos in range(n_positions):
            pos_variance = ensemble_variance[:, pos, :]  # (n_sequences, n_targets)

            position_stats[pos] = {
                'mean_variance': np.mean(pos_variance, axis=0),  # Per target
                'std_variance': np.std(pos_variance, axis=0),
                'max_variance': np.max(pos_variance, axis=0),
                'variance_percentiles': {
                    p: np.percentile(pos_variance, p, axis=0)
                    for p in [25, 50, 75, 90, 95]
                }
            }

            # Add reliability classification
            mean_var = np.mean(position_stats[pos]['mean_variance'])
            if pos < self.position_ranges['reliable'][1]:
                position_stats[pos]['reliability'] = 'high'
            elif pos < self.position_ranges['moderate'][1]:
                position_stats[pos]['reliability'] = 'moderate'
            else:
                position_stats[pos]['reliability'] = 'low'

        self.position_statistics = position_stats
        return position_stats

    def compute_position_aware_metrics(self, predictions, true_labels, sequence_lengths=None):
        """
        Compute validation metrics with sequence-length-aware position handling.
        """
        if sequence_lengths is None:
            sequence_lengths = self.sequence_lengths

        metrics = {}

        # Group sequences by length for consistent processing
        unique_lengths = np.unique(sequence_lengths)

        for seq_length in unique_lengths:
            length_mask = sequence_lengths == seq_length
            if not length_mask.any():
                continue

            # Extract sequences of this length
            length_predictions = predictions[length_mask]
            length_labels = true_labels[length_mask]

            # Truncate to actual sequence length
            actual_length = min(seq_length, predictions.shape[1])
            length_predictions = length_predictions[:, :actual_length, :]
            length_labels = length_labels[:, :actual_length, :]

            # Get adapted position ranges for this sequence length
            adapted_ranges = self.adapt_position_ranges_to_sequence_length(actual_length)

            # Compute metrics for each position range
            for range_name, (start_pos, end_pos) in adapted_ranges.items():
                if start_pos >= end_pos:  # Skip empty ranges
                    continue

                # Ensure range doesn't exceed actual sequence length
                end_pos = min(end_pos, actual_length)
                if start_pos >= end_pos:
                    continue

                range_preds = self.safe_position_range_slice(length_predictions, start_pos, end_pos, actual_length)
                range_labels = self.safe_position_range_slice(length_labels, start_pos, end_pos, actual_length)
                range_mask = ~np.isnan(range_labels)

                if range_mask.sum() > 0:
                    range_mse = mean_squared_error(
                        range_labels[range_mask],
                        range_preds[range_mask]
                    )
                    metric_key = f'{range_name}_rmse_len{seq_length}'
                    metrics[metric_key] = np.sqrt(range_mse)

        # Compute overall weighted metric across all sequence lengths
        weighted_scores = []
        weights = []

        for seq_length in unique_lengths:
            length_weight = np.sum(sequence_lengths == seq_length)  # Weight by frequency

            # Combine position-range scores for this length
            adapted_ranges = self.adapt_position_ranges_to_sequence_length(seq_length)
            length_score = 0.0
            range_weights = {'reliable': 0.6, 'moderate': 0.3, 'uncertain': 0.1}

            for range_name, weight in range_weights.items():
                metric_key = f'{range_name}_rmse_len{seq_length}'
                if metric_key in metrics:
                    length_score += weight * metrics[metric_key]

            if length_score > 0:
                weighted_scores.append(length_score)
                weights.append(length_weight)

        # Overall weighted score
        if weighted_scores:
            metrics['overall_weighted_rmse'] = np.average(weighted_scores, weights=weights)
        else:
            metrics['overall_weighted_rmse'] = float('inf')

        return metrics

    def safe_position_range_slice(self, array, start_pos, end_pos, max_length):
        """
        Safely slice array with position range validation.

        Args:
            array: Input array to slice
            start_pos: Start position
            end_pos: End position
            max_length: Maximum valid length

        Returns:
            Safely sliced array
        """
        # Validate and adjust range bounds
        start_pos = max(0, min(start_pos, max_length))
        end_pos = max(start_pos, min(end_pos, max_length))

        if start_pos >= end_pos:
            # Return empty slice with correct dimensions
            empty_shape = list(array.shape)
            empty_shape[1] = 0  # Zero length dimension
            return np.empty(empty_shape)

        return array[:, start_pos:end_pos, :]

    def adapt_position_ranges_to_sequence_length(self, sequence_length):
        """
        Dynamically adapt position ranges based on actual sequence length.

        Args:
            sequence_length: Actual sequence length (107 or 130)

        Returns:
            Adapted position ranges dictionary
        """
        if sequence_length <= 68:
            # Short sequences: single reliable range
            return {
                'reliable': (0, sequence_length),
                'moderate': (sequence_length, sequence_length),  # Empty range
                'uncertain': (sequence_length, sequence_length)  # Empty range
            }
        elif sequence_length <= 91:
            # Medium sequences: reliable + moderate
            return {
                'reliable': (0, 68),
                'moderate': (68, sequence_length),
                'uncertain': (sequence_length, sequence_length)  # Empty range
            }
        else:
            # Full sequences: all ranges active
            return {
                'reliable': (0, 68),
                'moderate': (68, 91),
                'uncertain': (91, min(sequence_length, 130))
            }

    def pseudo_labeling_corruption_analysis(self, pre_pl_metrics, post_pl_metrics,
                                            cv_metrics, public_lb_scores=None):
        """
        Analyze validation corruption after pseudo-labeling implementation.

        Args:
            pre_pl_metrics: Validation metrics before pseudo-labeling
            post_pl_metrics: Validation metrics after pseudo-labeling
            cv_metrics: Cross-validation metrics (potentially corrupted)
            public_lb_scores: Public leaderboard scores (if available)

        Returns:
            Dictionary with corruption analysis results
        """
        corruption_analysis = {
            'cv_reliability': 'unknown',
            'validation_shift': {},
            'recommended_strategy': 'unknown'
        }

        # Compute metric shifts
        for metric in pre_pl_metrics:
            if metric in post_pl_metrics:
                shift = post_pl_metrics[metric] - pre_pl_metrics[metric]
                corruption_analysis['validation_shift'][metric] = {
                    'absolute_change': shift,
                    'relative_change': shift / pre_pl_metrics[metric] if pre_pl_metrics[metric] != 0 else np.inf
                }

        # Assess CV reliability
        if public_lb_scores is not None and len(cv_metrics) > 0:
            # Compare CV trend with public LB trend
            cv_variance = np.var(list(cv_metrics.values()))

            if cv_variance > 0.1:  # High variance suggests corruption
                corruption_analysis['cv_reliability'] = 'corrupted'
                corruption_analysis['recommended_strategy'] = 'use_public_lb_as_primary_validation'
            else:
                corruption_analysis['cv_reliability'] = 'potentially_reliable'
                corruption_analysis['recommended_strategy'] = 'cautious_cv_with_lb_confirmation'
        else:
            corruption_analysis['cv_reliability'] = 'unknown'
            corruption_analysis['recommended_strategy'] = 'implement_holdout_validation'

        return corruption_analysis

    def implement_position_aware_early_stopping(self, model, val_dataloader, device,
                                                patience=10, min_delta=0.001,
                                                position_weights=None):
        """
        Early stopping with position-aware validation metrics.

        Args:
            model: Model being validated
            val_dataloader: Validation data loader
            device: Computation device
            patience: Early stopping patience
            min_delta: Minimum improvement threshold
            position_weights: Weights for different position ranges

        Returns:
            Dictionary with early stopping decision and metrics
        """
        if position_weights is None:
            # Winner's strategy: emphasize reliable positions
            position_weights = {
                'reliable': 0.6,  # High weight on positions 0-68
                'moderate': 0.3,  # Medium weight on positions 68-91
                'uncertain': 0.1  # Low weight on positions >91
            }

        model.eval()
        all_predictions = []
        all_labels = []
        all_sequence_lengths = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Extract data (assuming your existing dataloader structure)
                embeddings = batch['embedding'].to(device)
                bpps = batch['bpp'].to(device)
                src_mask = batch['src_mask'].to(device)
                labels = batch['labels'].to(device)

                # Additional features
                deltaG = batch['deltaG'].to(device)
                graph_dist = batch['graph_dist'].to(device)
                nearest_p = batch['nearest_p'].to(device)
                nearest_up = batch['nearest_up'].to(device)

                # Forward pass
                predictions = model(embeddings, bpps, src_mask, deltaG,
                                    graph_dist, nearest_p, nearest_up)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Extract actual sequence lengths from source mask
                if src_mask.dim() == 3:  # (batch, layers, seq_len)
                    actual_lengths = (src_mask[:, 0, :] > 0).sum(dim=1).cpu().numpy()
                else:  # (batch, seq_len)
                    actual_lengths = (src_mask > 0).sum(dim=1).cpu().numpy()
                all_sequence_lengths.extend(actual_lengths.tolist())

        # Concatenate all predictions and labels
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        sequence_lengths = np.array(all_sequence_lengths)

        # Compute position-aware metrics
        metrics = self.compute_position_aware_metrics(predictions, labels, sequence_lengths)

        # Compute weighted validation score using length-specific metrics
        weighted_score = 0.0
        total_weight = 0.0

        # Aggregate across all sequence lengths for each range
        for range_name, weight in position_weights.items():
            range_scores = []
            range_weights = []

            # Collect scores for this range across all sequence lengths
            for seq_length in np.unique(sequence_lengths):
                metric_key = f'{range_name}_rmse_len{seq_length}'
                if metric_key in metrics:
                    length_frequency = np.sum(sequence_lengths == seq_length)
                    range_scores.append(metrics[metric_key])
                    range_weights.append(length_frequency)

            # Compute weighted average for this range
            if range_scores:
                range_avg = np.average(range_scores, weights=range_weights)
                weighted_score += weight * range_avg
                total_weight += weight

        if total_weight > 0:
            weighted_score /= total_weight
        else:
            weighted_score = metrics.get('overall_weighted_rmse', float('inf'))

        # Store validation history
        self.validation_history['weighted_score'].append(weighted_score)
        self.validation_history['detailed_metrics'].append(metrics)

        # Early stopping decision
        best_score = min(self.validation_history['weighted_score'][:-1]) if len(
            self.validation_history['weighted_score']) > 1 else float('inf')
        improvement = best_score - weighted_score

        early_stopping_result = {
            'current_score': weighted_score,
            'best_score': best_score,
            'improvement': improvement,
            'should_stop': False,
            'patience_remaining': patience,
            'detailed_metrics': metrics
        }

        # Implement early stopping logic
        if improvement < min_delta:
            patience -= 1
            early_stopping_result['patience_remaining'] = patience
            if patience <= 0:
                early_stopping_result['should_stop'] = True
        else:
            # Reset patience on improvement
            early_stopping_result['patience_remaining'] = patience

        return early_stopping_result

    def generate_position_analysis_report(self, output_path='position_analysis_report.html'):
        """
        Generate comprehensive position analysis report.

        Args:
            output_path: Path to save HTML report
        """
        if not self.position_statistics:
            print("No position statistics available. Run compute_position_variance_profile first.")
            return

        # Create visualizations
        positions = list(self.position_statistics.keys())

        # Variance profile plot
        variance_data = {
            'position': positions,
            'mean_variance': [self.position_statistics[p]['mean_variance'].mean() for p in positions],
            'reliability': [self.position_statistics[p]['reliability'] for p in positions]
        }

        plt.figure(figsize=(12, 6))

        # Plot variance profile
        plt.subplot(1, 2, 1)
        colors = {'high': 'green', 'moderate': 'orange', 'low': 'red'}
        for reliability in ['high', 'moderate', 'low']:
            mask = np.array(variance_data['reliability']) == reliability
            if mask.any():
                plt.scatter(np.array(positions)[mask],
                            np.array(variance_data['mean_variance'])[mask],
                            c=colors[reliability], label=f'{reliability} reliability', alpha=0.7)

        plt.xlabel('Sequence Position')
        plt.ylabel('Mean Prediction Variance')
        plt.title('Position-Wise Prediction Variance Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add position range boundaries
        for range_name, (start, end) in self.position_ranges.items():
            plt.axvspan(start, end, alpha=0.1,
                        label=f'{range_name} range ({start}-{end})')

        # Validation history plot
        plt.subplot(1, 2, 2)
        if self.validation_history['weighted_score']:
            plt.plot(self.validation_history['weighted_score'], 'b-', linewidth=2)
            plt.xlabel('Validation Step')
            plt.ylabel('Weighted Validation Score')
            plt.title('Position-Aware Validation History')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save report
        plt.savefig(output_path.replace('.html', '_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Generate HTML report
        plots_path = output_path.replace('.html', '_plots.png')
        html_content = f"""<html>
<head><title>Position-Aware Validation Analysis Report</title></head>
<body>
<h1>Position-Aware Validation Analysis Report</h1>

<h2>Executive Summary</h2>
<p>Position-aware validation analysis reveals fundamental differences in prediction 
reliability across sequence positions, confirming winner's insight about positions >91.</p>

<h2>Position Reliability Classification</h2>
<ul>
    <li><strong>High Reliability (0-68):</strong> Competition-scored region with consistent predictions</li>
    <li><strong>Moderate Reliability (68-91):</strong> Pseudo-labeling eligible region</li>
    <li><strong>Low Reliability (>91):</strong> High variance region excluded from scoring</li>
</ul>

<h2>Validation Strategy Recommendations</h2>
<p>Based on pseudo-labeling corruption analysis:</p>
<ul>
    <li>Primary validation: Position-weighted metrics emphasizing reliable regions</li>
    <li>Secondary validation: Public leaderboard correlation monitoring</li>
    <li>Early stopping: Position-aware improvement tracking</li>
</ul>

<img src="{plots_path}" alt="Position Analysis Plots" style="max-width:100%;">

</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Position analysis report saved to: {output_path}")


# INTEGRATION: Enhanced validation wrapper for existing training pipeline
def integrate_position_aware_validation_into_training(train_function, val_dataloader,
                                                      sequence_lengths, device):
    """
    Wrapper to integrate position-aware validation into existing training pipeline.

    Args:
        train_function: Existing training function
        val_dataloader: Validation data loader
        sequence_lengths: Array of sequence lengths
        device: Computation device

    Returns:
        Enhanced training function with position-aware validation
    """
    validator = PositionAwareValidator(sequence_lengths)

    def enhanced_train_function(*args, **kwargs):
        # Execute original training
        result = train_function(*args, **kwargs)

        # Add position-aware validation analysis
        if 'model' in kwargs:
            model = kwargs['model']
            validation_result = validator.implement_position_aware_early_stopping(
                model, val_dataloader, device
            )

            # Enhance result with position-aware metrics
            result['position_aware_metrics'] = validation_result
            result['validation_strategy'] = 'position_aware'

        return result

    return enhanced_train_function