import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(15),
    v2.RandomApply([v2.GaussianBlur(3)], p=0.3),
    v2.ToTensor(),
])

class Config:
    batch_size = 32
    epochs = 10
    lr = 1e-3
    num_classes = 37

class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.LazyLinear(Config.num_classes)
        ) 

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer, epochs, scheduler):
    model.train()
    for epoch in range(epochs): 
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
        
            loss.backward()
            optimizer.step()
        scheduler.step()

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    training_data = datasets.OxfordIIITPet(
        root='data', 
        split='trainval',
        download=True,
        transform=transforms,
    )
    train_dataloader = DataLoader(training_data, batch_size=Config.batch_size, shuffle=True)

    model = PetClassifier().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr = Config.lr,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.01,
    )
    loss_fn = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs)
    train_loop(train_dataloader, model, loss_fn, optimizer, Config.epochs, scheduler)

    torch.save(model.state_dict(), "model.pth")