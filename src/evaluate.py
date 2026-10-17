
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import SimpleCNN

import matplotlib.pyplot as plt
import seaborn as sns
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_FILES = {
    "CNN without augmentation": MODEL_DIR / "best_cnn.pth",
    "CNN with augmentation": MODEL_DIR / "best_cnn_aug15.pth",
}

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
def evaluate_model(model, test_loader, device):
    """Collect predictions and calculate test accuracy."""
    model.eval()

    all_labels = []
    all_predictions = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

    accuracy = (
        sum(
            predicted == actual
            for predicted, actual in zip(all_predictions, all_labels)
        )
        / len(all_labels)
    )

    return all_labels, all_predictions, accuracy


def save_classification_report(labels, predictions, model_name):
    """Save per-class metrics to a CSV file."""
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    safe_name = model_name.lower().replace(" ", "_")
    report_path = OUTPUT_DIR / f"classification_report_{safe_name}.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1-score",
            "support",
        ])

        for class_name in CLASS_NAMES:
            metrics = report[class_name]
            writer.writerow([
                class_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"],
            ])

    print(f"Saved classification report: {report_path}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        batch_size=64,
        num_workers=0,
        val_size=5000,
        seed=42,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        model = SimpleCNN().to(device)
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        labels, predictions, accuracy = evaluate_model(
            model, test_loader, device
        )

        print(f"\n{'=' * 50}")
        print(model_name)
        print(f"Test Accuracy: {accuracy:.2%}")

        print("\nClassification Report:")
        print(
            classification_report(
                labels,
                predictions,
                labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )
        save_classification_report(labels, predictions, model_name)

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                labels,
                predictions,
                labels=list(range(len(CLASS_NAMES))),
            )
        )

        
        # Plot and save the confusion matrix
        cm = confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )
        save_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"

        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved confusion matrix: {save_path}")



if __name__ == "__main__":
    main()
