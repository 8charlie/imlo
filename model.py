"""ResNet-18 for Oxford-IIIT Pet (37 classes), plus the evaluation that train.py
and test.py share.

Layout follows the 18-layer network in He et al. 2016, "Deep Residual Learning
for Image Recognition" (https://arxiv.org/abs/1512.03385), Table 1: a 7x7
stride-2 stem and 3x3 max pool, then four stages of two basic blocks with 64,
128, 256 and 512 channels. Stages 2-4 halve the resolution in their first block
and project the shortcut with a 1x1 convolution + BN (option B in the paper).
Convolutions have no bias because each is followed by BatchNorm. Convolution
weights use He normal initialisation (He et al. 2015,
https://arxiv.org/abs/1502.01852).
"""

import torch
from torch import nn
from torchvision.transforms import v2

NUM_CLASSES = 37
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
# Evaluation averages the logits of two centre crops: resize to 248 and to 256, crop 224.
EVAL_RESIZES = (248, 256)


class BasicBlock(nn.Module):
    """Two 3x3 conv-BN layers added to an identity or projected shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.functional.relu(out + self.shortcut(x))


class PetClassifier(nn.Module):
    """ResNet-18. The final layer is named `classifier`: train.py gives it AdamW
    instead of Muon by that name."""

    def __init__(self, num_classes=NUM_CLASSES, widths=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        stages, in_channels = [], widths[0]
        for index, (width, count) in enumerate(zip(widths, blocks_per_stage)):
            blocks = [BasicBlock(in_channels, width, stride=1 if index == 0 else 2)]
            blocks += [BasicBlock(width, width) for _ in range(count - 1)]
            stages.append(nn.Sequential(*blocks))
            in_channels = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.classifier(self.pool(x).flatten(1))


class EvalViews:
    """Both normalised centre-crop views of one image, stacked: (2, 3, 224, 224)."""

    def __init__(self):
        self.views = [
            v2.Compose([
                v2.ToImage(),
                v2.Resize(size, antialias=True),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ])
            for size in EVAL_RESIZES
        ]

    def __call__(self, image):
        return torch.stack([view(image) for view in self.views])


def evaluate(device, dataloader, model):
    """Accuracy, averaging each image's logits over its EvalViews."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            count, views = X.shape[:2]
            logits = model(X.flatten(0, 1)).view(count, views, -1).mean(1)
            correct += (logits.argmax(1) == y).sum().item()
    return correct / len(dataloader.dataset)
