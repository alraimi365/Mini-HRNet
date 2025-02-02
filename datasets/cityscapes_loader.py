import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from datasets.augmentations import get_augmentations
from config_loader import ConfigLoader

# Load Configuration
config = ConfigLoader()

class CityscapesDataset(datasets.Cityscapes):
    """Custom Cityscapes dataset loader using PyTorch's built-in class."""
    def __init__(self, root: str, split: str = "train", mode: str = "fine", target_type="semantic"):
        super().__init__(root=root, split=split, mode=mode, target_type=target_type)

        # Load augmentation settings from config
        self.transforms = get_augmentations(
            split=split,
            image_size=tuple(config.get("dataset")["image_size"]),
            use_augmentations=config.get("dataset")["augmentations"]
        )

    def __getitem__(self, index: int):
        """Loads and applies transformations to an image-label pair."""
        img, target = super().__getitem__(index)
        return self.transforms(img), target

def load_cityscapes():
    """Loads Cityscapes dataset into DataLoader using settings from config."""
    dataset_config = config.get("dataset")
    
    train_set = CityscapesDataset(root=dataset_config["root"], split="train")
    val_set = CityscapesDataset(root=dataset_config["root"], split="val")
    test_set = CityscapesDataset(root=dataset_config["root"], split="test")

    return (
        DataLoader(train_set, batch_size=dataset_config["batch_size"], num_workers=dataset_config["num_workers"], shuffle=True),
        DataLoader(val_set, batch_size=dataset_config["batch_size"], num_workers=dataset_config["num_workers"], shuffle=False),
        DataLoader(test_set, batch_size=dataset_config["batch_size"], num_workers=dataset_config["num_workers"], shuffle=False),
    )
