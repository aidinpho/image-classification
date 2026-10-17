
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "best_resnet18_finetune.pth"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
BATCH_SIZE = 64
NUM_WORKERS = 0

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    # ResNet18 evaluation preprocessing
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # Load the CIFAR-10 test set
    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=eval_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test images: {len(test_dataset)}")

    # Rebuild the same ResNet18 architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    criterion = nn.CrossEntropyLoss()

    # Evaluate without calculating gradients
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                predictions == labels
            ).sum().item()
            total_images += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / total_images
    accuracy = total_correct / total_images

    print("\nTest Results")
    print("-" * 40)
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    report_path = (
        OUTPUT_DIR
        / "classification_report_resnet18_finetune.csv"
    )

    import pandas as pd

    pd.DataFrame(report).transpose().to_csv(
        report_path,
        index_label="class",
    )

    print(f"\nClassification report saved: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES))),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="ResNet18 Fine-tuning - Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    matrix_path = (
        OUTPUT_DIR
        / "confusion_matrix_resnet18_finetune.png"
    )
    fig.savefig(matrix_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved: {matrix_path}")

    # Save a compact summary
    summary = {
        "model": "ResNet18 fine-tuned",
        "test_images": total_images,
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
    }

    summary_path = (
        OUTPUT_DIR
        / "evaluation_summary_resnet18_finetune.json"
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Evaluation summary saved: {summary_path}")


if __name__ == "__main__":
    main()
