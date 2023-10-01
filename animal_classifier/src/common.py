import torch
from torch import nn
from torchvision import models, transforms


def network(name, classes, pretrained=False):
    if name not in {"cnn", "efficientnet"}:
        raise ValueError("Unknown architecture")
    if name == "efficientnet":
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, classes)
        return net
    # Небольшой baseline: он быстрее и проще, но у него нет преимущества
    # предобучения на ImageNet. Учитываем это при интерпретации сравнения.
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, classes),
    )


def transform(augment=False):
    ops = [transforms.Resize((224, 224))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    return transforms.Compose(
        ops
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = network(checkpoint["model"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint["classes"]
