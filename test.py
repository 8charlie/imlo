import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

test = datasets.OxfordIIITPet(
    'data', 'test')