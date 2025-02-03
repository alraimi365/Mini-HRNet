import torch
import torch.nn as nn
import torchvision.transforms
import numpy as np

def multi_scale_inference_v2(model, image, scales=[0.5,0.75,1.0,1.25,1.5,1.75], num_classes=19):
    """
    Multi-scale inference with horizontal flip and region-based averaging.

    Args:
        model (torch.nn.Module): Trained model.
        image (torch.Tensor): Input image tensor.
        scales (list): Scaling factors for multi-scale inference.

    Returns:
        torch.Tensor: Aggregated multi-scale predictions.
    """
    crop_size = (512, 1024)  # Reference size
    _, _, ori_height, ori_width = image.size()  # Original image dimensions

    trans_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
    stride_h = crop_size[0]
    stride_w = crop_size[1]
    
    final_pred = torch.zeros([1, num_classes, ori_height, ori_width]).cuda()
    
    for scale in scales:
        new_w = int(scale * ori_width)
        new_h = int(scale * ori_height)
        trans = torchvision.transforms.Resize(size=(new_h, new_w), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        new_img = trans(image)

        if scale <= 1.25:
            size = new_img.size()
            pred = model(new_img)

            flip_img = trans_flip(new_img)
            flip_output = model(flip_img)
            flip_pred = trans_flip(flip_output)
            
            preds = (pred + flip_pred) * 0.5  # Averaging predictions
            preds = nn.functional.interpolate(preds, size=size[-2:], mode='bilinear', align_corners=False)
            
        else:
            _, _, new_h, new_w = new_img.size()
            rows = int(np.ceil((new_h - crop_size[0]) / stride_h)) + 1
            cols = int(np.ceil((new_w - crop_size[1]) / stride_w)) + 1

            preds = torch.zeros([1, num_classes, new_h, new_w]).cuda()
            count = torch.zeros([1, 1, new_h, new_w]).cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + crop_size[0], new_h)
                    w1 = min(w0 + crop_size[1], new_w)
                    h0 = max(h1 - crop_size[0], 0)
                    w0 = max(w1 - crop_size[1], 0)
                    crop_img = new_img[:, :, h0:h1, w0:w1]

                    size = crop_img.size()
                    pred = model(crop_img)

                    flip_img = trans_flip(crop_img)
                    flip_output = model(flip_img)
                    flip_pred = trans_flip(flip_output)

                    pred = (pred + flip_pred) * 0.5
                    pred = nn.functional.interpolate(pred, size=size[-2:], mode='bilinear', align_corners=False)

                    preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                    count[:, :, h0:h1, w0:w1] += 1

            preds /= count

        preds = nn.functional.interpolate(preds, (ori_height, ori_width), mode='bilinear', align_corners=False)
        final_pred += preds
    
    return final_pred
