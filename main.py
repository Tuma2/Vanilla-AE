import torch
from torch import nn

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(inplace=True)
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(x)

        return decoded



# Define transformations (optional)
transform = transforms.Compose([transforms.ToTensor()])

# Load the dataset
train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Get images and labels for a specific example (optional)
train_image, train_label = train_data[0]

print("Image shape:", train_image.shape)
print("Label:", train_label)


data = torch.tensor(train_image,dtype=torch.float16r)

input_dim=data.shape[1]
encoding_dim=3

model = Autoencoder(input_dim,encoding_dim)
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters())
# Define a data loader for training data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

for epoch in range(100):
    for images, _ in train_loader:  # Ignore labels for now
        # Convert images to torch tensors
        images = images.float()

        # Forward pass
        output = model(images)
        loss = criterion(output, images)  # Reconstruct the input image

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Iteration:{epoch},Epoch: {epoch + 1}/{2000}, Loss: {loss.item():.4f}")

# Use the first batch of test data for visualization (optional)
test_images, _ = next(iter(torch.utils.data.DataLoader(test_data, batch_size=1)))
test_images = test_images.float()

encoded_data = model.encoder(test_images)
decoded_data = model.decoder(encoded_data)

# Print the original, encoded, and decoded data
print("Original Data:")
print(test_images)  # Use test data for visualization
print("\nEncoded Data:")
print(encoded_data)
print("\nDecoded Data:")
print(decoded_data)
#Issue if at line 50 look into it