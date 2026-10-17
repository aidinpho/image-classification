
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
EPOCHS = 15
VAL_SIZE = 5000
SEED = 42

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    # Keep early pretrained layers frozen and in evaluation mode.
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

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

    model = resnet18(weights=weights)

    # Freeze the entire network first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the classifier and unfreeze the final residual block.
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # A smaller learning rate for pretrained features.
    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable parameters:", trainable_params)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_resnet18_finetune.pth"
    history_path = OUTPUT_DIR / "training_history_resnet18_finetune.json"

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "layer4_lr": [],
        "fc_lr": [],
    }

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["layer4_lr"].append(optimizer.param_groups[0]["lr"])
        history["fc_lr"].append(optimizer.param_groups[1]["lr"])

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print("Best ResNet18 fine-tuned checkpoint saved.")

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Accuracy: {train_accuracy:.2%} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.2%} "
            f"| Layer4 LR: {optimizer.param_groups[0]['lr']:.2g} "
            f"| FC LR: {optimizer.param_groups[1]['lr']:.2g}"
        )

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    print("\nFinal ResNet18 fine-tuning results:")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print("Model saved to:", model_path)
    print("History saved to:", history_path)


if __name__ == "__main__":
    main()
