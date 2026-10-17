import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_DIR / "outputs" / "training_history.json"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main():
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.show()
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [value * 100 for value in history["train_accuracy"]],
        marker="o",
        label="Train Accuracy",
    )
    plt.plot(
        epochs,
        [value * 100 for value in history["val_accuracy"]],
        marker="o",
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=150)
    plt.show()
    plt.close()

    print("Plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()