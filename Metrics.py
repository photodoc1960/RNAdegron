import numpy as np
import torch

def accuracy(predictions,ground_truths):
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
    denom=np.sqrt((1+(N2-N1)/N3)*(1+(N1-N2)/N4))
    return (1-sens-spec)/denom
    
import torch


def weighted_mcrmse_tensor(preds: torch.Tensor,
                           targets: torch.Tensor,
                           errors: torch.Tensor,
                           mask_threshold: float = 10.0,
                           value_error_ratio: float = 1.5) -> torch.Tensor:
    """
    Enhanced MCRMSE with NaN handling and winner's column weights.
    preds, targets, errors: shape (B, L, 5) float tensors
    Returns weighted MCRMSE scalar tensor.
    """
    # Winner's column weights favoring scored columns
    col_weights = torch.tensor([0.3, 0.3, 0.3, 0.05, 0.05],
                               device=preds.device, dtype=preds.dtype)

    # Handle NaN values in targets (from position-level masking)
    nan_mask = ~torch.isnan(targets)

    # Additional position-level masking for high uncertainty
    additional_mask = torch.ones_like(targets)
    high_error_condition = (errors > mask_threshold) & (targets.abs() / errors < value_error_ratio)
    additional_mask = additional_mask.masked_fill(high_error_condition, 0.0)

    # Combine masks
    final_mask = nan_mask.float() * additional_mask

    # Compute squared errors only for valid positions
    valid_positions = final_mask > 0
    diff2 = torch.zeros_like(preds)
    diff2[valid_positions] = (preds[valid_positions] - targets[valid_positions]) ** 2

    # Sum over batch and length dimensions
    mse = diff2.sum(dim=(0, 1)) / (final_mask.sum(dim=(0, 1)) + 1e-8)  # shape (5,)
    rmse = torch.sqrt(mse)  # shape (5,)
    weighted = rmse * col_weights  # shape (5,)
    return weighted.sum()  # scalar