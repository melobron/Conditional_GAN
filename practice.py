import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

input = torch.randint(0, 10, (5,))
model = nn.Embedding(10, 10)
output = model(input)
print(output)

