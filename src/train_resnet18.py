
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SIZE = 5000
SEED = 42

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    # Keep the pretrained backbone in evaluation mode.
    # This also prevents its BatchNorm statistics from changing.
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()
    model.layer4.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # ImageNet normalization for pretrained ResNet18
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download/load CIFAR-10
    full_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=None,
    )

    train_size = len(full_dataset) - VAL_SIZE
    generator = torch.Generator().manual_seed(SEED)

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, VAL_SIZE],
        generator=generator,
    )

    # Use independent datasets so train/validation transforms differ.
    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=eval_transform,
    )
    train_subset.dataset = train_dataset
    val_subset.dataset = val_dataset

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    # Load pretrained weights. This may download weights on first run.
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained ResNet18 weights. "
            "Check your internet connection and try again."
        ) from exc

    # Freeze all pretrained parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final layer for CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    model = model.to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable parameters:", trainable_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_resnet18.pth"
    history_path = OUTPUT_DIR / "training_history_resnet18.json"

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print("Best ResNet18 checkpoint saved.")

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Accuracy: {train_accuracy:.2%} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.2%}"
        )

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    # Evaluate the best validation checkpoint on the test set.
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    print("\nFinal ResNet18 results:")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print("Model saved to:", model_path)
    print("History saved to:", history_path)


if __name__ == "__main__":
    main()
