#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:54:38 2023

@author: jakecastro
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse


### NOTE: change so that for every single epoch, test and train is ran.

#  ---------------  Dataset  ---------------

    
class HeartDataset(Dataset):
    """Heart Failure dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        df = pd.read_csv("/Users/jakecastro/Desktop/Classes/BME450/heart.csv")

        print(df.head())
        # Grouping variable names
        self.categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        self.target = "HeartDisease"

        # One-hot encoding of categorical variables
        self.encoded = pd.get_dummies(df)
        print(self.encoded)

        # Save target and predictors
        self.X = self.encoded.drop(self.target, axis=1)
        self.y = self.encoded[self.target]
        
    
    

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]


#  ---------------  Model  ---------------

class Net(nn.Module):

    def __init__(self, D_in, H, D_out = 1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, 5)
        self.fc5 = nn.Linear(5, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)

        return x.squeeze()


#  ---------------  Training  ---------------

def train(csv_file, n_epochs = 1000):
    """Trains the model.
    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """
    # Load dataset
    dataset = HeartDataset(csv_file)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size = 100, shuffle=True)
    testloader = DataLoader(testset, batch_size = 100, shuffle=True)

    print(trainloader.dataset)
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    D_in, H = 20, 9
    net = Net(D_in, H).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.0025, weight_decay=0.001)

    # Train the net
    loss_per_iter_train = []
    epoch_loss_train = []
    for epoch in range(n_epochs):

        running_loss_train = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss_train += loss.item()
            loss_per_iter_train.append(loss.item())

        epoch_loss_train.append(running_loss_train)
        
    # print(len(epoch_loss_train))
    # print(epoch_loss_train)
        
    # print(len(loss_per_batch_train))
    loss_per_iter_test = []
    epoch_loss_test = []
    
    for epoch in range(n_epochs):

        running_loss_test = 0.0
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss_test += loss.item()
            loss_per_iter_test.append(loss.item())

        epoch_loss_test.append(running_loss_test)
        
        

    # Comparing training to test
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs.float())
    print("Root mean squared error")
    print("Training:", np.sqrt(epoch_loss_train[-1]))
    print("Test", np.sqrt(epoch_loss_test[-1]))

    # Plot training loss curve
    #plt.plot(np.arange(len(loss_per_iter_train)), loss_per_iter_train, "-", alpha=0.5, label="Training Loss per epoch")
    plt.plot(np.arange(len(epoch_loss_train)), epoch_loss_train, "-", alpha=0.5, label="Training Loss per epoch")
    plt.plot(np.arange(len(epoch_loss_test)), epoch_loss_test, "-", alpha=0.5, label="Testing Loss per epoch")
    #plt.plot(np.arange(len(loss_per_iter_train), step=4) + 3, loss_per_batch_train, ".-", label="Loss per mini-batch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# By default, read csv file in the same directory as this script
csv_file = "/Users/jakecastro/Desktop/Classes/BME450/heart.csv"

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                    help="Dataset file used for training")
parser.add_argument("--epochs", "-e", type=int, nargs="?", default=1000, help="Number of epochs to train")
args = parser.parse_args()

# Call the main function of the script
train(args.file, args.epochs)

# OBSERVATIONS
# - # of epochs affects loss
# - Batch size affects loss
# - weight_decay affects loss
