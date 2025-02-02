from torchvision.transforms import v2
import torch

def get_augmentations(split: str, image_size=(512, 1024), use_augmentations=True):
    """Returns dataset-specific transformations."""
    
    if split == "train":
        transforms_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if use_augmentations:
            transforms_list += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(size=image_size),
            ]
        return v2.Compose(transforms_list)
    
    elif split in ["val", "test"]:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
