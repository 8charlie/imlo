import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
from torch.optim.lr_scheduler import OneCycleLR

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    #v2.RandomRotation(15),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

eval_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(232, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Config:
    batch_size = 32
    epochs = 30
    lr = 1e-2
    num_classes = 37

class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            #nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(512, Config.num_classes),
        ) 

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits

def eval_loop(device, dataloader, model, loss_fn):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).float().sum().item()
    avg_loss = loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

def train_loop(device, train_loader, model, loss_fn, optimizer, scheduler, epochs, eval_loader=None):
    best_eval_acc = 0
    for epoch in range(epochs): 
        model.train()
        avg_loss = 0
        correct = 0
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
        
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item()
            correct += (pred.argmax(1) == y).float().sum().item()
        print(f"--- Epoch {epoch+1} ---")
        train_acc = correct / len(train_loader.dataset)
        train_loss = avg_loss / len(train_loader)
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc * 100:.2f}%")
        if eval_loader is not None:
            eval_loss, eval_acc = eval_loop(device, eval_loader, model, loss_fn)
            print(f"eval loss: {eval_loss:.4f}, eval acc: {eval_acc * 100:.2f}%")
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save(model.state_dict(), "model.pth")

    if eval_loader is None:
        torch.save(model.state_dict(), "model.pth")
    else:
        print(f"best eval acc: {100 * best_eval_acc}")

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"device: {device}")

    train_full = datasets.OxfordIIITPet(
        root='data', 
        split='trainval',
        download=True,
        transform=train_transforms,
    )
    eval_full = datasets.OxfordIIITPet(
        root='data',
        split='trainval',
        download=True,
        transform=eval_transforms,
    )
    rng = np.random.default_rng(seed=42)
    indicies = rng.permutation(len(train_full))
    eval_size = int(0.1 * len(train_full))
    eval_indicies, train_indicies = indicies[:eval_size], indicies[eval_size:]
    train_set = Subset(train_full, train_indicies)
    eval_set = Subset(eval_full, eval_indicies)

    train_full_loader = DataLoader(train_full, batch_size=Config.batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=Config.batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

    model = PetClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr = Config.lr,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.005,
    )
    scheduler = OneCycleLR(optimizer, max_lr=Config.lr, epochs=Config.epochs, steps_per_epoch=len(train_full_loader))
    train_loop(device, train_full_loader, model, loss_fn, optimizer, scheduler, Config.epochs)