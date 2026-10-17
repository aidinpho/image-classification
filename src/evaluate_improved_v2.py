
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch import nn
from torchvision import transforms

from dataset import get_dataloaders, CIFAR10_MEAN, CIFAR10_STD
from model import ImprovedCNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_improved_cnn_v2.pth"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CLASS_NAMES = [
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


@torch.no_grad()
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    _, _, test_loader = get_dataloaders(
        eval_transform=eval_transform,
        batch_size=64,
        num_workers=0,
        val_size=5000,
        seed=42,
    )

    model = ImprovedCNN().to(device)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    all_labels = []
    all_predictions = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

    accuracy = (
        sum(
            true_label == prediction
            for true_label, prediction in zip(
                all_labels, all_predictions
            )
        )
        / len(all_labels)
    )

    print(f"\nTest Accuracy: {accuracy:.2%}")

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = OUTPUT_DIR / "classification_report_improved_v2.csv"
    report_df.to_csv(report_path)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title("ImprovedCNN V2 - CIFAR-10 Confusion Matrix")
    fig.tight_layout()

    matrix_path = OUTPUT_DIR / "confusion_matrix_improved_v2.png"
    fig.savefig(matrix_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSaved classification report to:", report_path)
    print("Saved confusion matrix to:", matrix_path)


if __name__ == "__main__":
    main()
