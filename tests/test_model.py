
import sys
from pathlib import Path

import pytest
import torch
from torchvision import models
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import SimpleCNN, ImprovedCNN


@pytest.mark.parametrize(
    "model_class",
    [SimpleCNN, ImprovedCNN],
)
def test_custom_model_output_shape(model_class):
    model = model_class()
    images = torch.randn(4, 3, 32, 32)

    with torch.inference_mode():
        outputs = model(images)

    assert outputs.shape == (4, 10)


def test_resnet18_output_shape():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        outputs = model(images)

    assert outputs.shape == (2, 10)


def test_resnet18_probabilities_sum_to_one():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_cifar10_class_names():
    expected_classes = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    }

    assert len(expected_classes) == 10
