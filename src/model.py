"""Model definitions for the CIFAR-10 classifier."""

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A small 3-block CNN, good enough as a baseline for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        """Build the conv feature extractor and the linear classifier head."""
        super().__init__()
        # Three conv+BN+ReLU+pool blocks, halving the spatial size each time:
        # 32x32 -> 16x16 -> 8x8 -> 4x4, while growing channels 3->32->64->128.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        # Flatten the 128x4x4 feature map and classify with a small MLP,
        # with dropout for regularization.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Run the input batch through the feature extractor, then the classifier head."""
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 inputs.

    The stock torchvision ResNet is meant for 224x224 ImageNet images, so
    the first conv is replaced with a 3x3 one and the initial max-pool is
    dropped. This is the usual trick for CIFAR-sized images.
    """
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    """Instantiate a model by name: 'simple_cnn' or 'resnet18'."""
    architecture = architecture.lower()
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if architecture == "resnet18":
        return build_resnet18(num_classes=num_classes)
    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        "Supported: simple_cnn, resnet18"
    )
