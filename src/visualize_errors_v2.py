
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from dataset import get_dataloaders, CIFAR10_MEAN, CIFAR10_STD
from model import ImprovedCNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_improved_cnn_v2.pth"

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


def denormalize(image):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    image = image.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


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

    # Collect errors for the cat <-> dog confusion
    cat_dog_errors = []
    other_errors = []

    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

        for i in range(len(labels)):
            true_label = labels[i].item()
            predicted_label = predictions[i].item()

            if true_label == predicted_label:
                continue

            item = (
                images[i].cpu(),
                true_label,
                predicted_label,
            )

            if (true_label, predicted_label) in [(3, 5), (5, 3)]:
                cat_dog_errors.append(item)
            else:
                other_errors.append(item)

    print("Cat predicted as dog:", sum(
        true == 3 and pred == 5
        for _, true, pred in cat_dog_errors
    ))
    print("Dog predicted as cat:", sum(
        true == 5 and pred == 3
        for _, true, pred in cat_dog_errors
    ))
    print("Other misclassified images:", len(other_errors))

    # Display up to 16 cat/dog errors
    samples = cat_dog_errors[:16]

    if not samples:
        print("No cat/dog errors found.")
    else:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()

        for ax, (image, true_label, predicted_label) in zip(
            axes, samples
        ):
            ax.imshow(denormalize(image))
            ax.set_title(
                f"True: {CLASS_NAMES[true_label]}\n"
                f"Pred: {CLASS_NAMES[predicted_label]}",
                fontsize=9,
            )
            ax.axis("off")

        for ax in axes[len(samples):]:
            ax.axis("off")

        fig.suptitle("ImprovedCNN V2 - Cat/Dog Errors")
        fig.tight_layout()
        plt.show()

    # Display up to 16 other errors
    samples = other_errors[:16]

    if samples:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()

        for ax, (image, true_label, predicted_label) in zip(
            axes, samples
        ):
            ax.imshow(denormalize(image))
            ax.set_title(
                f"True: {CLASS_NAMES[true_label]}\n"
                f"Pred: {CLASS_NAMES[predicted_label]}",
                fontsize=9,
            )
            ax.axis("off")

        for ax in axes[len(samples):]:
            ax.axis("off")

        fig.suptitle("ImprovedCNN V2 - Other Errors")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
