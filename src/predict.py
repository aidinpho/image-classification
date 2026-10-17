
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_resnet18_finetune.pth"

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

    results = []
    for probability, index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        results.append(
            f"{CLASSES[index]}: {probability * 100:.2f}%"
        )

    return results


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
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp",
                ),
                ("All files", "*.*"),
            ],
            parent=root,
        )

        if not image_path:
            print("No image selected. Exiting.")
            return

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = load_model(MODEL_PATH, device)
        results = predict_image(image_path, model, device)

        result_text = "Top 3 predictions:\n\n" + "\n".join(results)

        print(result_text)

        messagebox.showinfo(
            "Image Classification Results",
            result_text,
            parent=root,
        )

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
