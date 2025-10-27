import numpy as np
import torch


def accuracy(predictions, ground_truths):
    return np.sum(predictions==ground_truths)/len(ground_truths)
    
    
def sensitivity(predictions,ground_truths):
    '''
    Here it is assumed:
    0=negative
    1=positive
    '''
    return 1-len(predictions[(predictions==0)*(ground_truths==1)])/len(ground_truths[ground_truths==1])



def specificity(predictions,ground_truths):
    '''
    Here it is assumed:
    0=negative
    1=positive
    '''
    return 1-len(predictions[(predictions==1)*(ground_truths==0)])/len(ground_truths[ground_truths==0])
   
def MCC(predictions,ground_truths):
    '''
    Here it is assumed:
    0=negative
    1=positive
    '''
    N1=len(predictions[(predictions==0)&(ground_truths==1)])
    N2=len(predictions[(predictions==1)&(ground_truths==0)])
    N3=len(ground_truths[ground_truths==1])
    N4=len(ground_truths[ground_truths==0])
    sens=1-N1/N3
    spec=1-N2/N4
    denom = np.sqrt((1 + (N2 - N1) / N3) * (1 + (N1 - N2) / N4))
    return (1 - sens - spec) / denom


def weighted_mcrmse_tensor(preds: torch.Tensor,
                           targets: torch.Tensor,
                           errors: torch.Tensor,
                           mask_threshold: float = 10.0,
                           value_error_ratio: float = 1.5) -> torch.Tensor:
    """
    Enhanced MCRMSE with comprehensive NaN handling and winner's column weights.

    Args:
        preds: Predictions tensor (B, L, 5)
        targets: Target values tensor (B, L, 5)
        errors: Error weights tensor (B, L, 5)
        mask_threshold: Threshold for high error masking
        value_error_ratio: Ratio threshold for uncertainty masking

    Returns:
        Weighted MCRMSE scalar tensor
    """
    # Winner's column weights favoring scored columns
    col_weights = torch.tensor([0.3, 0.3, 0.3, 0.05, 0.05],
                               device=preds.device, dtype=preds.dtype)

    # Handle NaN values in both targets AND predictions
    nan_mask_targets = ~torch.isnan(targets)
    nan_mask_preds = ~torch.isnan(preds)
    nan_mask_errors = ~torch.isnan(errors)

    # Combined NaN mask - valid only if all three are not NaN
    nan_mask = nan_mask_targets & nan_mask_preds & nan_mask_errors

    # Additional position-level masking for high uncertainty
    # Only compute for valid (non-NaN) positions
    additional_mask = torch.ones_like(targets)
    safe_targets = torch.where(nan_mask, targets, torch.zeros_like(targets))
    safe_errors = torch.where(nan_mask, errors, torch.ones_like(errors))  # Use 1.0 to avoid division by zero

    high_error_condition = (errors > mask_threshold) & (safe_targets.abs() / (safe_errors + 1e-8) < value_error_ratio)
    additional_mask = additional_mask.masked_fill(high_error_condition, 0.0)

    # Combine masks
    final_mask = nan_mask.float() * additional_mask

    # Compute squared errors only for valid positions
    valid_positions = final_mask > 0
    diff2 = torch.zeros_like(preds)
    if valid_positions.any():
        diff2[valid_positions] = (preds[valid_positions] - targets[valid_positions]) ** 2

    # Sum over batch and length dimensions
    mask_sum = final_mask.sum(dim=(0, 1))
    # Avoid division by zero - if no valid positions, return large penalty
    mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))

    mse = diff2.sum(dim=(0, 1)) / mask_sum  # shape (5,)
    rmse = torch.sqrt(mse + 1e-10)  # Add epsilon for numerical stability
    weighted = rmse * col_weights  # shape (5,)
    return weighted.sum()  # scalar