
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_resnet18_finetune.pth"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def load_model(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(
        probabilities, k=3
    )

    # Keep each class index paired with its correct probability.
    results = [
        (CLASSES[int(index)], float(probability) * 100)
        for index, probability in zip(
            top_indices.cpu().tolist(),
            top_probabilities.cpu().tolist(),
        )
    ]

    return image.copy(), results


def show_results(image, results, image_path):
    labels = [name.title() for name, _ in results]
    scores = [score for _, score in results]

    fig, (ax_image, ax_chart) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )

    fig.suptitle(
        "CIFAR-10 Image Classification | ResNet18",
        fontsize=16,
        fontweight="bold",
    )

    # Display the original image.
    ax_image.imshow(image)
    ax_image.set_title("Input Image", fontsize=13)
    ax_image.axis("off")

    # Display top-3 predicted classes.
    positions = list(range(len(labels)))
    bars = ax_chart.barh(positions, scores, height=0.58)

    ax_chart.set_yticks(positions)
    ax_chart.set_yticklabels(labels, fontsize=11)
    ax_chart.invert_yaxis()
    ax_chart.set_xlim(0, 120)
    ax_chart.set_xlabel("Softmax score (%)")
    ax_chart.set_title("Top 3 Predictions", fontsize=13)
    ax_chart.grid(axis="x", linestyle="--", alpha=0.3)
    ax_chart.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        ax_chart.text(
            min(score + 1.5, 101),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}%",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    best_class, best_score = results[0]

    fig.text(
    0.5, 0.075,
    f"Predicted class: {best_class.title()} ({best_score:.2f}%)",
    ha="center",
    fontsize=13,
    fontweight="bold",
    )

    fig.text(
        0.5, 0.025,
        f"Image: {Path(image_path).name}",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.10, 1, 0.92))
    plt.show()


def main():
    root = tk.Tk()
    root.withdraw()

    try:
        if not MODEL_PATH.is_file():
            messagebox.showerror(
                "Model not found",
                f"Could not find the trained model:\n{MODEL_PATH}",
                parent=root,
            )
            return

        image_path = filedialog.askopenfilename(
            title="Select an image to classify",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
            parent=root,
        )

        if not image_path:
            return

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = load_model(MODEL_PATH, device)
        image, results = predict_image(image_path, model, device)

        print("Top 3 predictions:")
        for name, score in results:
            print(f"{name}: {score:.2f}%")

        show_results(image, results, image_path)

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            f"Could not classify the selected image.\n\n{error}",
            parent=root,
        )
        raise

    finally:
        root.destroy()


if __name__ == "__main__":
    main()
