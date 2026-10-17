
import torch
from torch import nn

from dataset import get_dataloaders
from model import SimpleCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, _ = get_dataloaders(batch_size=64)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # Save the initial weights for comparison
    first_parameter = next(model.parameters())
    weights_before = first_parameter.detach().clone()

    # Training step
    model.train()
    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    weights_after = first_parameter.detach()

    print("Loss:", loss.item())
    print(
        "Gradient exists:",
        first_parameter.grad is not None,
    )
    print(
        "Weights changed:",
        not torch.equal(weights_before, weights_after),
    )


if __name__ == "__main__":
    main()
