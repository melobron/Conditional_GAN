import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import random
import time
import matplotlib.pyplot as plt

from model import Generator, Discriminator


# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Random Seed
seed = random.randint(1, 10000)
torch.manual_seed(seed)

# Parameters
batch_size = 16
nEpochs = 10
lr = 0.0001
z_dim = 100

number = 3

# Model
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator'))

z = torch.randn(batch_size, z_dim, device=device)
label = torch.tensor((number,)).to(device)
labels = torch.cat([label for _ in range(batch_size)], dim=0)
fake_images = generator((z, labels))

# Visualization
plt.imshow(make_grid(fake_images.cpu(), normalize=True).permute(1, 2, 0))
plt.show()

