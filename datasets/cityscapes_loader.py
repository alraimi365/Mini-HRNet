import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as F
from datasets.augmentations import get_augmentations
from config_loader import ConfigLoader

# Cityscapes Label Mapping Based on `Cityscapes_modified`
CITYSCAPES_LABEL_MAP = {
    0: 19,     # Unlabeled (ignored)
    1: 19,     # Ego vehicle (ignored)
    2: 19,     # Rectification border (ignored)
    3: 19,     # Out of ROI (ignored)
    4: 19,     # Static (ignored)
    5: 19,     # Dynamic (ignored)
    6: 19,     # Ground (ignored)
    7: 0,      # Road
    8: 1,      # Sidewalk
    9: 19,     # Parking (ignored)
    10: 19,    # Rail track (ignored)
    11: 2,     # Building
    12: 3,     # Wall
    13: 4,     # Fence
    14: 19,    # Guard rail (ignored)
    15: 19,    # Bridge (ignored)
    16: 19,    # Tunnel (ignored)
    17: 5,     # Pole
    18: 19,    # Polegroup (ignored)
    19: 6,     # Traffic light
    20: 7,     # Traffic sign
    21: 8,     # Vegetation
    22: 9,     # Terrain
    23: 10,    # Sky
    24: 11,    # Person
    25: 12,    # Rider
    26: 13,    # Car
    27: 14,    # Truck
    28: 15,    # Bus
    29: 19,    # Caravan (ignored)
    30: 19,    # Trailer (ignored)
    31: 16,    # Train
    32: 17,    # Motorcycle
    33: 18,    # Bicycle
    -1: 19     # License plate (ignored)
}

def remap_labels(label_tensor):
    """
    Remaps Cityscapes labels to match the expected class IDs in `Cityscapes_modified`.
    """
    label_mapped = label_tensor.clone()
    for key, value in CITYSCAPES_LABEL_MAP.items():
        label_mapped[label_tensor == key] = value
    return label_mapped


def cityscapes_collate_fn(batch):
    """
    Custom collate function to convert images and labels to PyTorch tensors.
    """
    images, labels = zip(*batch)

    # Convert images to tensors
    images = [F.to_tensor(img) if isinstance(img, np.ndarray) else img for img in images]

    # Convert labels to tensors and apply remapping
    labels = [remap_labels(torch.tensor(np.array(lbl), dtype=torch.long)).squeeze(0) for lbl in labels]

    return torch.stack(images), torch.stack(labels)

def load_cityscapes(split="train"):
    """Loads Cityscapes dataset into DataLoader using settings from config."""
    config = ConfigLoader()
    dataset_config = config.get("dataset")

    # Define dataset paths and transformations
    cityscapes_root = dataset_config["root"]
    image_size = tuple(dataset_config["image_size"])
    use_augmentations = dataset_config["augmentations"]

    # Get image & label transformations
    transform_fn, target_transform_fn = get_augmentations(split=split, image_size=image_size, use_augmentations=use_augmentations)

    # Load dataset with correct transforms
    dataset = datasets.Cityscapes(
        root=cityscapes_root,
        split=split,
        mode="fine",
        target_type="semantic",
        transform=transform_fn,          # For images
        target_transform=target_transform_fn,  # For labels
    )

    # Return DataLoader with correct split
    return DataLoader(
        dataset,
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
        collate_fn=cityscapes_collate_fn,
        shuffle=True if split == "train" else False,
    )
