import torch
import torch.nn as nn
from config_loader import ConfigLoader

# Load config
config = ConfigLoader()
weight_option = config.get("training")["weight_option"]

NUM_CLASSES = 19
IGNORE_VALUE = 19

# Predefined Weight Options
WEIGHTS = {
    "hrnet": torch.FloatTensor([
        0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
        0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
        1.0865, 1.1529, 1.0507
    ]).cuda(),
    
    "sqrt_freq": torch.FloatTensor([
        0.0069, 0.0169, 0.0087, 0.0514, 0.0445, 0.0376, 0.0914, 0.0561,
        0.0104, 0.0387, 0.0208, 0.0377, 0.1133, 0.0157, 0.0805, 0.0859,
        0.0863, 0.1326, 0.0647
    ]).cuda(),

    "inv_freq": torch.FloatTensor([
        0.0006, 0.0046, 0.0013, 0.0403, 0.0285, 0.0208, 0.0890, 0.0385,
        0.0020, 0.0211, 0.0103, 0.0171, 0.1023, 0.0034, 0.1130, 0.1040,
        0.0888, 0.2570, 0.0573
    ]).cuda()
}

def get_loss_function():
    """Returns the configured CrossEntropyLoss with selected weights."""
    weight = WEIGHTS.get(weight_option, None)
    if weight_option != "none":
        print(f"✅ Using CrossEntropyLoss with {weight_option} weights")
    else:
        print(f"✅ Using CrossEntropyLoss with no weights")
    
    return nn.CrossEntropyLoss(ignore_index=IGNORE_VALUE, weight=weight)
