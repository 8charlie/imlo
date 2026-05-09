import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from train import PetClassifier, Config # REMOVE THIS BEFORE SUBMISSION

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).float().sum().item()

    loss /= num_batches
    correct /= size
    print(f"Accuracy = {100 * correct}%")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    testing_data = datasets.OxfordIIITPet(
        root='data', 
        split='test',
        download=True,
        transform=transforms,
    )
    num_classes = len(testing_data.classes)
    test_dataloader = DataLoader(testing_data, batch_size=Config.batch_size, shuffle=True)

    model = PetClassifier().to(device)
    model.load_state_dict(torch.load("model.pth"))
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr = Config.lr,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.01,
    )
    loss_fn = nn.CrossEntropyLoss()
    test_loop(test_dataloader, model, loss_fn)

