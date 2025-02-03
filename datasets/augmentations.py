from torchvision.transforms import v2
import torch

def get_augmentations(split: str, image_size=(512, 1024), use_augmentations=True):
    """Returns transformations separately for images and labels for Cityscapes dataset."""
    
    # **Image Transformations (transform)**
    image_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    
    # **Label Transformations (target_transform)**
    label_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=False),  # Labels should remain uint8 for class IDs
    ]
    
    if split == "train":
        if use_augmentations:
            aug_transforms = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(size=image_size),
            ]
            image_transforms.extend(aug_transforms)
            label_transforms.extend(aug_transforms)  # Apply same augmentations to labels
        
        # Apply normalization to images (NOT labels)
        image_transforms.append(
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    elif split in ["val", "test"]:
        image_transforms.extend([
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    return v2.Compose(image_transforms), v2.Compose(label_transforms)
