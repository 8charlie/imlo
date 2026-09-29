import math
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torch.optim.lr_scheduler import OneCycleLR

from model import NORMALIZE_MEAN, NORMALIZE_STD, EvalViews, PetClassifier, evaluate


class Config:
    seed = 1
    batch_size = 16
    epochs = 30
    lr = 6e-3
    weight_decay = 0.02
    label_smoothing = 0.1
    grad_clip = 5.0
    warmup = 0.3  # fraction of steps spent raising the learning rate
    muon = True  # False: AdamW for every parameter
    num_workers = min(8, os.cpu_count() or 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Horizontal flips are applied in train_loop (alternating flip), not here.
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(224, scale=(0.35, 1.0), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.RandomRotation(15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


class IndexedDataset(Dataset):
    """Returns (image, label, index) so train_loop knows which images to flip."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, index


def orthogonalise(matrix, steps=5):
    """Quintic Newton-Schulz iteration approximating the polar factor U V^T of a
    matrix, with the coefficients from Jordan et al. 2024 (see Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """Muon (https://kellerjordan.github.io/posts/muon/), written
    from the published equations: Nesterov momentum, then each update is
    orthogonalised so every direction gets a comparable step. Convolution filters are
    flattened to (out, in * kh * kw) matrices; torch.optim.Muon only accepts 2D weights.
    Updates are scaled by 0.2 * sqrt(max(rows, cols)) to match AdamW's update size
    (Liu et al. 2025, https://arxiv.org/abs/2502.16982), so both optimisers can share
    one learning rate and weight decay."""

    def __init__(self, params, lr, weight_decay, momentum=0.9):
        super().__init__(
            params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for weight in group["params"]:
                if weight.grad is None:
                    continue
                buffer = self.state[weight].setdefault(
                    "momentum", torch.zeros_like(weight)
                )
                buffer.lerp_(weight.grad, 1 - beta)
                update = weight.grad.lerp(buffer, beta).flatten(1)  # Nesterov
                scale = 0.2 * math.sqrt(max(update.shape))
                weight.mul_(1 - lr * group["weight_decay"])
                weight.add_(orthogonalise(update).view_as(weight), alpha=-lr * scale)


def build_optimizers(model):
    """Muon for the convolution weights, AdamW for the rest (BatchNorm, classifier).
    With Config.muon off, AdamW for every parameter."""
    if not Config.muon:
        return [
            torch.optim.AdamW(
                model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
            )
        ]
    convs = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]
    conv_ids = {id(weight) for weight in convs}
    others = [p for p in model.parameters() if id(p) not in conv_ids]
    return [
        Muon(convs, lr=Config.lr, weight_decay=Config.weight_decay),
        torch.optim.AdamW(others, lr=Config.lr, weight_decay=Config.weight_decay),
    ]


def train_loop(
    device, train_loader, model, loss_fn, optimizers, schedulers, flip_parity
):
    learning_rates = []
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X, y, index in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Alternating flip (Jordan 2024, https://arxiv.org/abs/2404.00498): each image is
            # mirrored on every other epoch from a random starting parity, instead of at random.
            flip = (flip_parity[index] ^ (epoch % 2)).bool().to(device)
            X = torch.where(flip[:, None, None, None], X.flip(-1), X)
            pred = model(X)
            loss = loss_fn(pred, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (pred.argmax(1) == y).sum().item()
        print(f"--- Epoch {epoch + 1} ---")
        train_acc = correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader.dataset)
        print(
            f"train loss: {train_loss:.4f}, train acc: {train_acc * 100:.2f}%",
            flush=True,
        )
        learning_rates.append(f"{schedulers[0].get_last_lr()[0]:.2e}")
    return learning_rates


if __name__ == "__main__":
    set_seed(Config.seed)
    torch.backends.cudnn.benchmark = True

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    train_data = IndexedDataset(
        datasets.OxfordIIITPet(
            root="data",
            split="trainval",
            download=True,
            transform=train_transforms,
        )
    )
    train_loader = DataLoader(
        train_data,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=Config.num_workers > 0,
        generator=torch.Generator().manual_seed(Config.seed),
    )

    model = PetClassifier().to(device)
    optimizers = build_optimizers(model)
    # One identical schedule per optimiser. OneCycleLR also cycles the momentum:
    # Muon's `momentum` and AdamW's first beta.
    schedulers = [
        OneCycleLR(
            optimizer,
            max_lr=Config.lr,
            total_steps=Config.epochs * len(train_loader),
            pct_start=Config.warmup,
            final_div_factor=1e4,
        )
        for optimizer in optimizers
    ]
    loss_fn = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
    flip_parity = torch.randint(
        0,
        2,
        (len(train_data),),
        generator=torch.Generator().manual_seed(Config.seed + 1),
    )

    learning_rates = train_loop(
        device, train_loader, model, loss_fn, optimizers, schedulers, flip_parity
    )
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "model.pth")
    print(f"\n---Learning rates---\n {learning_rates}")

    # Accuracy of the saved model on the full training dataset
    eval_data = datasets.OxfordIIITPet(
        root="data", split="trainval", download=True, transform=EvalViews()
    )
    eval_loader = DataLoader(
        eval_data, batch_size=64, shuffle=False, num_workers=Config.num_workers
    )
    train_acc = evaluate(device, eval_loader, model)
    print(f"\nAccuracy of saved model on full training dataset: {train_acc * 100:.2f}%")
