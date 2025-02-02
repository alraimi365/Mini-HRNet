import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config_loader import ConfigLoader

# Load Config
config = ConfigLoader()

# Dataset Settings
dataset_root = config.get("dataset")["root"]
batch_size = config.get("dataset")["batch_size"]
num_workers = config.get("dataset")["num_workers"]

# Model Settings
model_name = config.get("model")["name"]
pretrained = config.get("model")["pretrained"]

# Training Settings
epochs = config.get("training")["epochs"]
learning_rate = config.get("training")["learning_rate"]
loss_function = config.get("training")["loss_function"]
optimizer_name = config.get("training")["optimizer"]

# Logging Settings
log_dir = config.get("logging")["log_dir"]
save_model_dir = config.get("logging")["save_model_dir"]

# Dummy Dataset (Replace with Cityscapes later)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

# Dummy Model (Replace with HRNet later)
model = nn.Linear(10, 2)

# Select Loss Function
loss_fn = nn.CrossEntropyLoss()

# Select Optimizer
if optimizer_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training Loop (Basic)
print(f"Training {model_name} for {epochs} epochs...")

for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(inputs.view(inputs.size(0), -1))  # Flatten input

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training Complete!")
