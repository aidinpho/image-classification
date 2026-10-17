
import matplotlib.pyplot as plt
import torch

from dataset import get_dataloaders


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def main():
    train_loader, _ = get_dataloaders(batch_size=64)

    images, labels = next(iter(train_loader))
    class_names = train_loader.dataset.classes

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for index, ax in enumerate(axes.flat):
        # Reverse normalization for visualization
        image = images[index] * STD + MEAN
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(class_names[labels[index].item()])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
