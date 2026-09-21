import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import EvalViews, PetClassifier, evaluate

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    testing_data = datasets.OxfordIIITPet(
        root='data',
        split='test',
        download=True,
        transform=EvalViews(),
    )
    test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=False, num_workers=min(8, os.cpu_count() or 1))

    model = PetClassifier().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))

    test_acc = evaluate(device, test_dataloader, model)
    print(f"Accuracy of model on test dataset: {test_acc * 100:.2f}%")
