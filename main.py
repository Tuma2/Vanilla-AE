import csv

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

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


# readfile seperates the birthrate and life expectancy
# the sets them as float values and stores the value
# int the relevant arrays

def readfile(filename):
    data = []
    birthrate = []
    life_expectancy = []

    with open(filename, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        next(read)
        for row in read:
            data.append(row)
            birthrate.append(float(row[1]))
            life_expectancy.append(float(row[2]))

    return birthrate, life_expectancy


def writefile(filename, data):
    with open(filename, "a") as file:
        for line in data:
            file.write(str(line) + "\n")


birthrate, life_expectancy = readfile('data2008.csv')

data_Matrix = np.vstack([birthrate, life_expectancy]).T
data_Matrix = (data_Matrix - data_Matrix.mean()) / data_Matrix.std()
data_tensor = torch.from_numpy(data_Matrix).float()

model = Autoencoder(input_dim=2, encoding_dim=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 50000

for epoch in range(epochs):

    outputs = model(data_tensor)
    loss = criterion(outputs, data_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Iteration:{epoch},Epoch: {epoch + 1}/{20000}, Loss: {loss.item():.4f}")

encoded_data = model.encoder(data_tensor)
decoded_data = model.decoder(encoded_data)

# Print the original, encoded, and decoded data
print("Original Data:")
print(data_tensor)
print("\nEncoded Data:")
print(encoded_data)
print("\nDecoded Data:")
print(decoded_data)