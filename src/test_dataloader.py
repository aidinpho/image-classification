import torch

from dataset import get_dataloaders


def main():
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=64,
        num_workers=0,
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000
    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    print("DataLoader test passed!")


if __name__ == "__main__":
    main()
