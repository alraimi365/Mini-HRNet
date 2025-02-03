import numpy as np
import json
import torch

def compute_iou(confmat):
    """
    Compute per-class IoU from a confusion matrix.
    
    Args:
        confmat (torchmetrics.ConfusionMatrix): The confusion matrix computed during validation.
    
    Returns:
        dict: Per-class IoU values and mean IoU.
    """
    mat = confmat.compute()
    mat = mat.cpu().numpy()

    pos = np.sum(mat, axis=1)  # Ground truth counts per class
    res = np.sum(mat, axis=0)  # Predictions per class
    tp = np.diag(mat)  # True Positives

    iou_per_class = tp / np.maximum(1.0, pos + res - tp)  # IoU Calculation
    mean_iou = np.nanmean(iou_per_class)  # Mean IoU

    return {"per_class_iou": iou_per_class.tolist(), "mean_iou": mean_iou}


def save_iou_results(iou_dict, filename="iou_results.json"):
    """
    Save IoU results to a JSON file.
    
    Args:
        iou_dict (dict): Dictionary containing per-class and mean IoU.
        filename (str): File path to save results.
    """
    with open(filename, "w") as f:
        json.dump(iou_dict, f, indent=4)
    print(f"📁 IoU results saved to {filename}")
