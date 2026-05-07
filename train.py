import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

training_data = datasets.OxfordIIITPet(
    root='root', 
    split='trainval',
    download=True,
    transform=transform,
)

batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 37),
        ) 

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, epochs):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs): 
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
        
            loss.backward()
            optimizer.step()

model = NeuralNetwork().to(device)
optimizer = optim.AdamW(
    model.parameters(),
    lr = 1e-4,
    betas = (0.9, 0.999),
    eps = 1e-8,
    weight_decay = 0.01,
)
loss_fn = nn.CrossEntropyLoss()

train(train_dataloader, model, loss_fn, optimizer, epochs=5)