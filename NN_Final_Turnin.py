"""
@author: Jake Castro and Toby Miller
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
from sklearn.preprocessing import MinMaxScaler


#  ---------------  Dataset  --------------- 
# HeartDataset Class reads in CSV file, separates output from inputs, encodes
# categorical data, normalizes numerical data.

class HeartDataset(Dataset):
    def __init__(self, csv_file):

        mms = MinMaxScaler() # Normalization function
        
        df = pd.read_csv("/Users/jakecastro/Desktop/Classes/BME450/heart.csv")

        print(df.head())
        
        # Grouping variable names
        self.categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        self.numerical = ['Oldpeak', 'Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        self.target = "HeartDisease"

        # Encoding of categorical variables
        self.encoded = pd.get_dummies(df)
        print(self.encoded)
        
        # Normalization of Numerical Data
        self.encoded['Oldpeak'] = mms.fit_transform(self.encoded[['Oldpeak']])
        self.encoded['Age'] = mms.fit_transform(self.encoded[['Age']]) 
        self.encoded['RestingBP'] = mms.fit_transform(self.encoded[['RestingBP']])
        self.encoded['Cholesterol'] = mms.fit_transform(self.encoded[['Cholesterol']])
        self.encoded['MaxHR'] = mms.fit_transform(self.encoded[['MaxHR']])
        
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
## Net Class establishes structure of neural network

class Net(nn.Module):

    def __init__(self, D_in, H, D_out = 1):
        super().__init__()
        self.il = nn.Linear(D_in, H) # input layer
        self.hl1 = nn.Linear(H, H) # hidden layer
        self.hl2 = nn.Linear(H, H) # hidden layer
        self.hl3 = nn.Linear(H, H) # hidden layer
        self.ol = nn.Linear(H, D_out) # output layer
        self.relu = nn.ReLU() # nonlinearity function used
        # self.fc6 = nn.Linear(H, H)
        # self.fc7 = nn.Linear(H,H)
        # self.fc8 = nn.Linear(H, H)
        # self.fc9 = nn.Linear(H, D_out)
        # self.tanh = nn.Tanh() 


    def forward(self, x):
        x = self.il(x)
        x = self.relu(x)
        x = self.hl1(x)
        x = self.relu(x)
        x = self.hl2(x)
        x = self.relu(x)
        x = self.hl3(x)
        x = self.relu(x)
        x = self.ol(x)
        # x = self.relu(x)
        # x = self.fc6(x)
        # x = self.relu(x)
        # x = self.fc7(x)
        # x = self.relu(x)
        # x = self.fc8(x)
        # x = self.relu(x)
        # x = self.fc9(x)

        return x.squeeze()


#  ---------------  Training and Testing ---------------

def train_test(csv_file, n_epochs = 100):

    # Load dataset
    dataset = HeartDataset(csv_file)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
    testloader = DataLoader(testset, batch_size = 32, shuffle = False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    D_in, H = 20, 9
    
    net = Net(D_in, H).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.00025, weight_decay=0.001)

    # Train the net
    epoch_loss_train = []
    epoch_loss_test = []
    accuracy = []
    
    for epoch in range(n_epochs):

        running_loss_train = 0.0
        running_loss_test = 0.0
        correct = 0
        
        net.train()
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

        epoch_loss_train.append(running_loss_train)
        
        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward + backward + optimize
                outputs = net(inputs.float())
                loss = criterion(outputs, labels.float())
                
                result = (outputs  > 0.5).float()

                # Save loss to plot
                running_loss_test = loss.item() * inputs.size(0)
                correct += (result == labels).float().sum()
                
              
            epoch_loss_test.append(running_loss_test)
            
        acc = 100 * (correct / len(testset))
        accuracy.append(acc)
        
        print("Epoch {} Accuracy: {}%".format(epoch + 1, acc ))
        

    # Comparing training to test
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs.float())
    print("\nRoot mean squared error")
    print("Training:", np.sqrt(epoch_loss_train[-1]))
    print("Test", np.sqrt(epoch_loss_test[-1]))
    

    # Plot training loss curve
    
    plt.plot(np.arange(0 , 1000, 1), accuracy, "-", label = "Accuracy Per Epoch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(len(epoch_loss_train)), np.sqrt(epoch_loss_train), "-", alpha=0.5, label="Training Loss per epoch")
    plt.plot(np.arange(len(epoch_loss_test)), np.sqrt(epoch_loss_test), "-", alpha=0.5, label="Testing Loss per epoch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    
################################################################################

## Executable Code

csv_file = "/Users/jakecastro/Desktop/Classes/BME450/heart.csv" # Directory location of csv file

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                    help="Dataset file used for training")
parser.add_argument("--epochs", "-e", type=int, nargs="?", default = 1000, help="Number of epochs to train")
args = parser.parse_args()

# Call the main function of the script
train_test(args.file, args.epochs)
