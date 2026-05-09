import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    #v2.Resize((232)),
    #v2.RandomCrop,
    v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    #v2.RandomRotation(15),
    #v2.RandomApply([v2.GaussianBlur(3)], p=0.3),
    #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.TrivialAugmentWide(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Config:
    batch_size = 32
    epochs = 30
    lr = 1e-3
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
            nn.LazyLinear(256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, Config.num_classes),
        ) 

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs): 
        avg_loss = 0
        correct = 0
        total = 0
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
        
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            correct += (pred.argmax(1) == y).float().sum().item()
            total += y.size(0)
        print(f"Epoch {epoch+1} avg loss: {avg_loss / len(dataloader)}, train acc: {100 * correct / total}")

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"device: {device}")

    training_data = datasets.OxfordIIITPet(
        root='data', 
        split='trainval',
        download=True,
        transform=transforms,
    )
    train_dataloader = DataLoader(training_data, batch_size=Config.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    model = PetClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr = Config.lr,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.01,
    )
    train_loop(train_dataloader, model, loss_fn, optimizer, Config.epochs)

    torch.save(model.state_dict(), "model.pth")