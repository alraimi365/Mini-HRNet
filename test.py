import torch
import numpy as np
from datasets.cityscapes_loader import load_cityscapes
from models.model import get_model
from config_loader import ConfigLoader
from models.losses import get_loss_function
from metrics.iou import compute_iou, save_iou_results
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import Accuracy
from inference.normal import normal_inference
from inference.multi_scale import multi_scale_inference_v2
import matplotlib.pyplot as plt

def main():
    # Load Configuration
    config = ConfigLoader()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    print("🔄 Loading dataset...")
    val_loader= load_cityscapes(split="val")
    print("✅ Validation dataset loaded successfully!")

    # Print Dataset Info
    val_samples = len(val_loader.dataset)
    sample_batch = next(iter(val_loader))  # Extract one sample

    print(f"📊 Validation Samples: {val_samples}")
    print(f"🖼️ Sample Batch Shape: {sample_batch[0].shape} (Images), {sample_batch[1].shape} (Labels)")

    # Load Model
    print("🔄 Loading model...")
    model = get_model()
    model.to(device)
    print("✅ Model loaded successfully!")

    # Load loss function dynamically
    loss_function = get_loss_function()

    # Initialize Metrics
    confmat = MulticlassConfusionMatrix(num_classes=config.get("dataset")["num_classes"], ignore_index=config.get("dataset")["ignore_value"]).to(device)
    accuracy_metric = Accuracy(task="multiclass", num_classes=config.get("dataset")["num_classes"], ignore_index=config.get("dataset")["ignore_value"]).to(device)

    # Get inference mode
    inference_mode = config.get("test")["inference_mode"]
    if inference_mode == "multi_scale":
        inference_function = multi_scale_inference_v2
        print("🔄 Using multi-scale inference.")
    else:
        inference_function = normal_inference
        print("🔄 Using normal inference.")

    # Run Evaluation
    print("\n🔍 Running evaluation...")
    results = evaluate(model, val_loader, loss_function, confmat, accuracy_metric, device, config, inference_function)
    print("\n✅ Evaluation Complete!")
    
    print(f"📊 Mean IoU: {results['iou']['mean_iou'] * 100:.2f}%")
    print(f"📊 Accuracy: {results['accuracy'] * 100:.2f}%")

    # Save IoU results
    save_iou_results(results['iou'])

# ------------------------- EVALUATION FUNCTION -------------------------

def evaluate(model, data_loader, loss_function, confmat, accuracy_metric, device, config, inference_function):
    """
    Runs inference on the test dataset and calculates loss, IoU, and accuracy.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): The test dataset loader.
        loss_function (torch.nn.Module): The loss function.
        confmat (torchmetrics.ConfusionMatrix): Confusion matrix for IoU calculation.
        accuracy_metric (torchmetrics.Accuracy): Accuracy metric.
        device (str): Device (CPU/GPU).
        config (ConfigLoader): Configuration settings.
        inference_function (function): The selected inference function (normal or multi-scale).

    Returns:
        dict: Contains losses, confusion matrix, accuracy, and IoU results.
    """
    model.eval()
    confmat.reset()
    accuracy_metric.reset()
    
    losses = []
    total_samples = len(data_loader.dataset)

    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            # Apply the selected inference function
            pred = inference_function(model, X)

            # Compute loss
            loss = loss_function(pred, y)
            losses.append(loss.item())

            # Update metrics
            confmat.update(pred, y)
            accuracy_metric.update(pred, y)

            # Logging every 'print_freq' batches
            if (i + 1) % config.get("test")["print_freq"] == 0:
                print(f"\n📝 Batch {i + 1}/{total_samples}")
                print(f"🔹 Loss: {np.nanmean(losses):.4f}")
                print(f"🔹 Mean IoU: {compute_iou(confmat)['mean_iou'] * 100:.2f}%")
                print(f"🔹 Accuracy: {accuracy_metric.compute().item() * 100:.2f}%")

    # Compute final IoU results
    iou_results = compute_iou(confmat)

    return {
        "losses": losses,
        "confmat": confmat,
        "accuracy": accuracy_metric.compute().item(),
        "iou": iou_results,
    }

# ------------------------- RUN SCRIPT -------------------------

if __name__ == "__main__":
    main()
