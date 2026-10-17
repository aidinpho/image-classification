import matplotlib.pyplot as plt
from torchvision import datasets
import numpy as np
import torch
from torchvision import transforms

def main():
    # Download CIFAR-10
    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
    )

    # Dataset information
    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Number of classes:", len(train_dataset.classes))
    print("Classes:", train_dataset.classes)

    # Display image dimensions
    image, label = train_dataset[0]

    print("Image size:", image.size)
    print("Image mode:", image.mode)
    print("First image label:", label)
    print("First image class:", train_dataset.classes[label])

    # Display 10 sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for index, ax in enumerate(axes.flat):
        image, label = train_dataset[index]

        print("Image array shape:", np.array(image).shape)

        ax.imshow(image)
        ax.set_title(train_dataset.classes[label])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    image, label = train_dataset[0]

    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)

    print("PIL image size:", image.size)
    print("Tensor shape:", image_tensor.shape)
    print("Tensor dtype:", image_tensor.dtype)
    print("Minimum pixel value:", image_tensor.min().item())
    print("Maximum pixel value:", image_tensor.max().item())
    print("Class:", train_dataset.classes[label])
    
    # Normalize CIFAR-10 using its commonly used channel statistics
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_image = normalize(image_tensor)

    print("\nAfter normalization:")
    print("Shape:", normalized_image.shape)
    print("Minimum:", normalized_image.min().item())
    print("Maximum:", normalized_image.max().item())
    print("Channel means:", normalized_image.mean(dim=(1, 2)))


if __name__ == "__main__":
    main()
