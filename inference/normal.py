import torch

def normal_inference(model, image):
    """
    Runs standard single-scale inference.

    Args:
        model (torch.nn.Module): Trained model.
        image (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Model predictions.
    """
    with torch.no_grad():
        pred = model(image)
    return pred
