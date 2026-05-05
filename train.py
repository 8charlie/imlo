import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.OxfordIIITPet(
    root='root', 
    train=True,
    #split='trainval',
    download=True,
    transform=ToTensor(),
)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size)

# Using GPU or CPU?
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() 