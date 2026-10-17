
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

HISTORY_FILES = {
    "CNN without augmentation": OUTPUT_DIR / "training_history.json",
    "CNN with augmentation": OUTPUT_DIR / "training_history_aug15.json",
}


def load_history(file_path):
    """Load training history from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_metric(histories, metric, title, ylabel, filename):
    """Plot one metric for both models."""
    plt.figure(figsize=(10, 6))

    for model_name, history in histories.items():
        values = history[metric]
        epochs = range(1, len(values) + 1)

        plt.plot(epochs, values, marker="o", label=model_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 16))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def main():
    histories = {}

    for model_name, file_path in HISTORY_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"History file not found: {file_path}"
            )

        histories[model_name] = load_history(file_path)

    for model_name, history in histories.items():
        print(f"\n{model_name}")
        print(f"Epochs: {len(history['train_loss'])}")

    plot_metric(
        histories,
        metric="train_accuracy",
        title="Training Accuracy Comparison",
        ylabel="Accuracy (%)",
        filename="comparison_train_accuracy.png",
    )

    plot_metric(
        histories,
        metric="val_accuracy",
        title="Validation Accuracy Comparison",
        ylabel="Accuracy (%)",
        filename="comparison_val_accuracy.png",
    )

    plot_metric(
        histories,
        metric="train_loss",
        title="Training Loss Comparison",
        ylabel="Loss",
        filename="comparison_train_loss.png",
    )

    plot_metric(
        histories,
        metric="val_loss",
        title="Validation Loss Comparison",
        ylabel="Loss",
        filename="comparison_val_loss.png",
    )


if __name__ == "__main__":
    main()
