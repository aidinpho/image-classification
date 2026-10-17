
import torch

from model import SimpleCNN, ImprovedCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    images = torch.randn(4, 3, 32, 32, device=device)

    models = {
        "SimpleCNN": SimpleCNN(),
        "ImprovedCNN": ImprovedCNN(),
    }

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(images)

        num_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print(f"\n{name}")
        print(f"Output shape: {tuple(outputs.shape)}")
        print(f"Trainable parameters: {num_parameters:,}")
        print(f"Device: {device}")

        assert outputs.shape == (4, 10)


if __name__ == "__main__":
    main()
