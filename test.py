import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

transforms = v2.Compose([
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
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, Config.num_classes),
        ) 

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits

def test_loop(device, dataloader, model, loss_fn):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).float().sum().item()
    accuracy = correct / len(dataloader.dataset)
    avg_loss = loss / len(dataloader)
    print(f"test loss = {avg_loss:.4f}, test acc = {100 * (accuracy):.2f}%")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    testing_data = datasets.OxfordIIITPet(
        root='data', 
        split='test',
        download=True,
        transform=transforms,
    )
    num_classes = len(testing_data.classes)
    test_dataloader = DataLoader(testing_data, batch_size=Config.batch_size, shuffle=False, num_workers=0)

    model = PetClassifier().to(device)
    model.load_state_dict(torch.load("model.pth"))
    
    loss_fn = nn.CrossEntropyLoss()
    test_loop(device, test_dataloader, model, loss_fn)