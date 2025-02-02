import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from datasets.cityscapes_loader import load_cityscapes
from models.model import get_model
from config_loader import ConfigLoader
from models.losses import get_loss_function

def main():
    # Load Configuration
    config = ConfigLoader()

    # Load Dataset
    print("🔄 Loading dataset...")
    train_loader, val_loader, test_loader = load_cityscapes()
    print("✅ Dataset loaded successfully!")

    # Print Dataset Info
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    sample_batch = next(iter(train_loader))  # This line caused the error
    print(f"📊 Training Samples: {train_samples}, Validation Samples: {val_samples}")
    print(f"🖼️ Sample Batch Shape: {sample_batch[0].shape} (Images), {sample_batch[1].shape} (Labels)")

    # Load Model
    print("🔄 Loading model...")
    model = get_model()
    print("✅ Model loaded successfully!")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print Model Summary
    print("\n📌 Model Summary:")
    summary(model, (3, 512, 1024))

    # Load loss function dynamically
    loss_function = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=config.get("training")["learning_rate"])

    # Placeholder Training Loop
    print("\n🔹 Placeholder Training Loop (To be implemented)...")
    for epoch in range(config.get("training")["epochs"]):
        print(f"Epoch {epoch+1}/{config.get('training')['epochs']} - Training... (placeholder)")

    print("🚀 Model & dataset loaded successfully. Training loop to be implemented.")

# Ensure the script runs correctly on Windows
if __name__ == '__main__':
    main()
    