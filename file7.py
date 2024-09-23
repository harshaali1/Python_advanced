# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from torchvision.models import inception_v3
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import os

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
latent_dim = 100  # Dimension of the latent space
image_size = 64  # Size of the generated images
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
eval_frequency = 10  # Evaluate image quality every 10 epochs

# Define the Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is a latent vector (e.g., 100 x 1 x 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Upsample to 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Upsample to 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Upsample to 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Output layer: 3 channels for RGB image
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is a 64x64x3 image
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample to 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample to 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample to 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer: Single value for real/fake classification
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze()

# Function to calculate Fréchet Inception Distance (FID)
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Initialize the Generator and Discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Initialize optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Load your dataset (replace with your actual dataset loading code)
class YourDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Example data loading for demonstration:
data = torch.randn(1000, 3, 64, 64)  # Replace with your actual data
dataset = YourDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, real_images in enumerate(dataloader):
        # Train the Discriminator
        ## Real Images
        real_images = real_images.to(device)
        real_labels = torch.ones(real_images.size(0), 1).to(device)  # Real labels = 1
        discriminator.zero_grad()
        output = discriminator(real_images)
        loss_D_real = criterion(output, real_labels.squeeze())

        ## Fake Images
        noise = torch.randn(real_images.size(0), latent_dim, 1, 1).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device)  # Fake labels = 0
        output = discriminator(fake_images.detach())
        loss_D_fake = criterion(output, fake_labels.squeeze())

        # Total Discriminator loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        # Train the Generator
        generator.zero_grad()
        output = discriminator(fake_images)
        loss_G = criterion(output, real_labels.squeeze())  # Train Generator to fool Discriminator
        loss_G.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')

    # Evaluate image quality every 'eval_frequency' epochs
    if (epoch + 1) % eval_frequency == 0:
        # Generate synthetic images
        with torch.no_grad():
            noise = torch.randn(64, latent_dim, 1, 1).to(device)
            generated_images = generator(noise)

        # Save generated images
        os.makedirs("generated_images", exist_ok=True)
        save_image(make_grid(generated_images.detach().cpu(), nrow=8, normalize=True),
                   f"generated_images/epoch_{epoch+1}.png")

        # ... (Add your image quality evaluation code here) ...

# Save the trained Generator model
torch.save(generator.state_dict(), "generator_model.pth")
