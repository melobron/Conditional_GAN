import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
batch_size = 2
nEpochs = 30
lr = 0.0002
z_dim = 100

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

mnist_train = MNIST(root='MNIST_data', train=True, transform=transform, download=True)
mnist_test = MNIST(root='MNIST_data', train=False, transform=transform, download=True)

train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# Model
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# Initialize weights
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

# Loss function and Optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Train
D_losses = []
G_losses = []


def train(epoch):
    for batch, (images, labels) in enumerate(train_dataloader):
        real_images, labels = images.to(device), labels.to(device)
        batch_size = real_images.shape[0]
        z = torch.randn(batch_size, z_dim, device=device)
        fake_images = generator((z, labels))

        # Discriminator: maximize log(D(x)) + log(1-D(G(z)))
        discriminator.zero_grad()
        real_result = discriminator((real_images, labels))
        real_target = torch.ones_like(real_result, device=device)

        fake_result = discriminator((fake_images.detach(), labels))
        fake_target = torch.zeros_like(fake_result, device=device)

        errorD_real = criterion(real_result, real_target)
        errorD_fake = criterion(fake_result, fake_target)
        errorD = errorD_real + errorD_fake
        errorD.backward()
        optimizerD.step()

        P_real = real_result.mean().item()
        P_fake = fake_result.mean().item()  # discriminator가 진짜라고 판별할 확률

        # Generator: maximize log(D(G(z)))
        generator.zero_grad()

        fake_images = generator((z, labels))
        fake_result = discriminator((fake_images, labels))
        fake_label = torch.ones_like(fake_result, device=device)
        errorG = criterion(fake_result, fake_label)
        errorG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
              % (epoch, nEpochs, batch + 1, len(train_dataloader),
                 errorD.item(), errorG.item()))

        G_losses.append(errorG.item())
        D_losses.append(errorD.item())


start = time.time()
for epoch in range(1, nEpochs+1):
    train(epoch)

print("time: ", time.time()-start)

# Save Model
torch.save(generator.state_dict(), 'generator')
torch.save(discriminator.state_dict(), 'discriminator')

# Loss Visualization
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('GDloss.png')

