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
    num_workers = min(8, os.cpu_count() or 1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Horizontal flips are applied in train_loop (alternating flip), not here.
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    v2.RandomRotation(15),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

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
    matrix, with the coefficients from Jordan et al. 2024 (see MuonWithAdamW)."""
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

class MuonWithAdamW(torch.optim.Optimizer):
    """Muon for parameter groups with muon=True, AdamW for the rest, both with
    decoupled weight decay. Written from the published equations:
    - Muon: Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/
      (Nesterov momentum, then the update matrix is orthogonalised so every
      direction gets a comparable step size);
    - AdamW: Loshchilov & Hutter 2019, https://arxiv.org/abs/1711.05101;
    - Muon updates are scaled by 0.2*sqrt(max(rows, cols)) so their RMS matches a
      typical AdamW update (Liu et al. 2025, https://arxiv.org/abs/2502.16982),
      letting both share one learning rate and weight decay.
    betas[0] is the momentum for both, so OneCycleLR can cycle it.
    """

    def __init__(self, param_groups, lr, weight_decay, betas=(.9, .999), eps=1e-8):
        super().__init__(param_groups, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, muon=False))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, decay, (beta1, beta2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                parameter.mul_(1 - lr * decay)
                if group["muon"]:
                    momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                    momentum.lerp_(grad, 1 - beta1)
                    direction = grad.lerp(momentum, beta1).reshape(len(grad), -1)
                    scale = .2 * math.sqrt(max(direction.shape))
                    parameter.add_(orthogonalise(direction).view_as(parameter), alpha=-lr * scale)
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denominator = (state["exp_avg_sq"] / (1 - beta2 ** state["step"])).sqrt_().add_(eps)
                    parameter.addcdiv_(state["exp_avg"], denominator, value=-lr / (1 - beta1 ** state["step"]))

def build_optimizer(model):
    """Muon for the convolution weights; AdamW for the classifier, BatchNorm and biases."""
    convs, others = [], []
    for name, parameter in model.named_parameters():
        is_conv = parameter.ndim >= 2 and not name.startswith("classifier")
        (convs if is_conv else others).append(parameter)
    return MuonWithAdamW([dict(params=convs, muon=True), dict(params=others)],
                         lr=Config.lr, weight_decay=Config.weight_decay)

def train_loop(device, train_loader, model, loss_fn, optimizer, scheduler, flip_parity):
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (pred.argmax(1) == y).sum().item()
        print(f"--- Epoch {epoch+1} ---")
        train_acc = correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader.dataset)
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc * 100:.2f}%", flush=True)
        learning_rates.append(f"{optimizer.param_groups[0]['lr']:.2e}")
    return learning_rates

if __name__ == "__main__":
    set_seed(Config.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"device: {device}")

    train_data = IndexedDataset(datasets.OxfordIIITPet(
        root='data',
        split='trainval',
        download=True,
        transform=train_transforms,
    ))
    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True,
                              num_workers=Config.num_workers, pin_memory=True,
                              persistent_workers=Config.num_workers > 0,
                              generator=torch.Generator().manual_seed(Config.seed))

    model = PetClassifier().to(device)
    optimizer = build_optimizer(model)
    scheduler = OneCycleLR(optimizer, max_lr=Config.lr, total_steps=Config.epochs * len(train_loader),
                           pct_start=Config.warmup, final_div_factor=1e4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
    flip_parity = torch.randint(0, 2, (len(train_data),), generator=torch.Generator().manual_seed(Config.seed + 1))

    learning_rates = train_loop(device, train_loader, model, loss_fn, optimizer, scheduler, flip_parity)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "model.pth")
    print(f"\n---Learning rates---\n {learning_rates}")

    # Accuracy of the saved model on the full training dataset, evaluated exactly like test.py
    eval_data = datasets.OxfordIIITPet(root='data', split='trainval', download=True, transform=EvalViews())
    eval_loader = DataLoader(eval_data, batch_size=64, shuffle=False, num_workers=Config.num_workers)
    train_acc = evaluate(device, eval_loader, model)
    print(f"\nAccuracy of saved model on full training dataset: {train_acc * 100:.2f}%")
