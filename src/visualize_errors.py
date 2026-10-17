
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import get_dataloaders
from model import SimpleCNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_FILES = {
    "Without augmentation": MODEL_DIR / "best_cnn.pth",
    "With augmentation": MODEL_DIR / "best_cnn_aug15.pth",
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

MEAN = torch.tensor(
    (0.4914, 0.4822, 0.4465)
).view(3, 1, 1)

STD = torch.tensor(
    (0.2470, 0.2435, 0.2616)
).view(3, 1, 1)


@torch.no_grad()
def get_predictions(model, test_loader, device):
    """Return true labels and predictions for the test set."""
    model.eval()
    all_labels = []
    all_predictions = []

    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

    return all_labels, all_predictions


def show_examples(
    dataset,
    labels,
    predictions_without,
    predictions_with,
    indices,
    title,
    max_images=12,
):
    """Display examples from one error category."""
    selected_indices = indices[:max_images]

    print(f"\n{title}: {len(indices)} images")

    if not selected_indices:
        print("No examples in this category.")
        return

    ncols = 4
    nrows = (len(selected_indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 3.8 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, index in zip(axes, selected_indices):
        image, true_label = dataset[index]

        # Undo normalization for visualization.
        image = (image * STD + MEAN).clamp(0, 1)
        image = image.permute(1, 2, 0)

        pred_without = predictions_without[index]
        pred_with = predictions_with[index]

        ax.imshow(image)
        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Without aug: {CLASS_NAMES[pred_without]}\n"
            f"With aug: {CLASS_NAMES[pred_with]}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(selected_indices):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


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

    predictions_by_model = {}
    labels = None

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

        current_labels, predictions = get_predictions(
            model, test_loader, device
        )

        if labels is None:
            labels = current_labels

        predictions_by_model[model_name] = predictions

        accuracy = sum(
            pred == actual
            for pred, actual in zip(predictions, current_labels)
        ) / len(current_labels)

        print(f"{model_name} accuracy: {accuracy:.2%}")

    predictions_without = predictions_by_model[
        "Without augmentation"
    ]
    predictions_with = predictions_by_model[
        "With augmentation"
    ]

    both_wrong = []
    augmentation_fixed = []
    augmentation_regressed = []

    for index, true_label in enumerate(labels):
        pred_without = predictions_without[index]
        pred_with = predictions_with[index]

        if pred_without != true_label and pred_with != true_label:
            both_wrong.append(index)
        elif pred_without != true_label and pred_with == true_label:
            augmentation_fixed.append(index)
        elif pred_without == true_label and pred_with != true_label:
            augmentation_regressed.append(index)

    print("\nError category summary")
    print("-" * 35)
    print(f"Both models wrong: {len(both_wrong)}")
    print(f"Fixed by augmentation: {len(augmentation_fixed)}")
    print(f"Correct before, wrong after: {len(augmentation_regressed)}")

    dataset = test_loader.dataset

    show_examples(
        dataset, labels,
        predictions_without, predictions_with,
        augmentation_fixed,
        "Augmentation Fixed: Baseline Wrong, Augmented Correct",
    )

    show_examples(
        dataset, labels,
        predictions_without, predictions_with,
        augmentation_regressed,
        "Augmentation Regressed: Baseline Correct, Augmented Wrong",
    )

    show_examples(
        dataset, labels,
        predictions_without, predictions_with,
        both_wrong,
        "Both Models Wrong",
    )


if __name__ == "__main__":
    main()
