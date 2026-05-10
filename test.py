import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from train import PetClassifier, Config # REMOVE THIS BEFORE SUBMISSION

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(232, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    test_dataloader = DataLoader(testing_data, batch_size=Config.batch_size, shuffle=False)

    model = PetClassifier().to(device)
    model.load_state_dict(torch.load("model.pth"))
    
    loss_fn = nn.CrossEntropyLoss()
    test_loop(device, test_dataloader, model, loss_fn)