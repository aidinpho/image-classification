
import torch
from torch import nn

from dataset import get_dataloaders
from model import SimpleCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, _ = get_dataloaders(batch_size=64)

    # Get one batch of real images
    images, labels = next(iter(train_loader))

    # Create the model
    model = SimpleCNN().to(device)

    # Move data to the same device as the model
    images = images.to(device)
    labels = labels.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    outputs = model(images)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Show results
    print("Input shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Output shape:", outputs.shape)
    print("First 5 predictions:", outputs.argmax(dim=1)[:5].tolist())
    print("First 5 actual labels:", labels[:5].tolist())
    print("Initial loss:", loss.item())


if __name__ == "__main__":
    main()
