import numpy as np
import pandas as pd
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

train = torchvision.datasets.OxfordIIITPet('root', 'trainval')
X = []
y = []

for image, label in train:
    X.append(image)
    y.append(label)