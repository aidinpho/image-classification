
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_DIR / "outputs" / "training_history_improved.json"
OUTPUT_DIR = PROJECT_DIR / "outputs"


with open(HISTORY_PATH, "r", encoding="utf-8") as file:
    history = json.load(file)

print("History keys:", history.keys())
print("Number of epochs:", len(history["train_loss"]))

epochs = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs, history["train_loss"], label="Train Loss")
axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
axes[0].set_title("ImprovedCNN - Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    epochs, history["train_accuracy"], label="Train Accuracy"
)
axes[1].plot(
    epochs, history["val_accuracy"], label="Validation Accuracy"
)
axes[1].set_title("ImprovedCNN - Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

output_path = OUTPUT_DIR / "improved_cnn_training_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {output_path}")
