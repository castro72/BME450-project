#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:45:14 2023

@author: jakecastro
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('/Users/jakecastro/Desktop/Classes/BME450/heart.csv')
df.head()

cat_threshold = 10
print(f'Variables with less than {cat_threshold} unique values and their values:\n')

# Some are binary features, since they are just a few I will encode them anyway if strings
for x in df.columns:
    if len(df[x].unique()) <= cat_threshold:
        print(x, np.unique(df[x].values, return_counts=True))
        

encoded_df = pd.get_dummies(df)
encoded_df.head()

## removes "HeartDisease Variable from table"
# features = [x for x in df.columns if x not in 'HeartDisease']
# print(len(features))

print(encoded_df)

# Looking at the categorical variables
dummy_cols = encoded_df.columns
categorical_cols = [x for x in df.columns if x not in dummy_cols]
print(f'Categorical variables encoded: {len(categorical_cols)}\n{categorical_cols}')

# looking at numerical variables
num_vars_cols = [x for x in df.columns if x not in categorical_cols]
print(f'Numerical variables encoded: {len(num_vars_cols)}\n{num_vars_cols}')
print(num_vars_cols)

# looking at numerical variables
num_vars_df = df[[x for x in df.columns if x not in categorical_cols]]


corr_matrix = num_vars_df.corr()
mask_matrix = np.triu(corr_matrix) # upper triangle of the matrix

sns.heatmap(corr_matrix, 
            vmin=corr_matrix.min().min(),
            vmax=corr_matrix.replace(1.0, -1).max().max(), # excluding the diagonal of ones from max
            mask = mask_matrix, # masking symmetric part of the corr matrix
            cmap="PiYG" # to visualize the extremities better
           )


target_col = 'HeartDisease'
features_col = [x for x in encoded_df.columns if x != target_col]

# Fixing a type for pytorch
X = encoded_df[features_col].values.astype('float32')
y = encoded_df[target_col].astype('int64')

# splitting train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# split validation from training set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)


class HeartDiseaseDataset(Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        target = self.labels.iloc[idx]
        return features, target
    
    
    # setting up the neural network (random seed set to 42 for reproducibility)
class HeartDiseaseModel(nn.Module):
    
    def __init__(self):
        torch.manual_seed(42)
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(len(features_col), 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    
    # selecting the number of epochs
num_epochs = 45

# instantiating the dataset and dataloader for both training and validation
heart_training = HeartDiseaseDataset(X_train, y_train)
heart_validating = HeartDiseaseDataset(X_val, y_val)

heart_trainloader = DataLoader(heart_training, batch_size=16, shuffle=True)
heart_valloader = DataLoader(heart_validating, batch_size=16, shuffle=True)

# instantiating the model, loss function, and optimizer
model = HeartDiseaseModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# to store the loss
loss_list = list()

# training the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(heart_trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    model.eval()
        
    # initialize the validation loss
    val_loss = 0.0
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the validation data
        for i, data in enumerate(heart_valloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        
    train_loss = running_loss/len(heart_training)
    val_loss = val_loss/len(heart_validating)
    loss_list.append((train_loss, val_loss))
    
    str_epoch = f'Epoch [{epoch+1}/{num_epochs}]'
    str_train_loss = f'Training Loss: {train_loss:.4f}'
    str_val_loss = f'Validation Loss: {val_loss:.4f}'
    print(f'{str_epoch}, {str_train_loss}, {str_val_loss}')
    
    
    # Evaluate the model on the validation set
model.eval()

with torch.no_grad():
    output = model.forward(torch.from_numpy(X_val))

# Convert the output to a readable format
predicted_labels = torch.argmax(output, dim=1)
print(classification_report(y_val, predicted_labels))

# I just separate the loss stored at each epoch
training_loss = np.array(loss_list)[:,0]
validation_loss = np.array(loss_list)[:,1]

plt.plot(training_loss, )
plt.plot(validation_loss)

skip = 5 # show a epoch after 5 skip
epoch_list = [x-1 if x > 0 else 0 for x in range(num_epochs+1) if x % skip == 0]
plt.xticks(ticks = epoch_list, labels = [x+1 for x in epoch_list])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Evaluate the model on the test set
model.eval()

with torch.no_grad():
    output = model.forward(torch.from_numpy(X_test))

# Convert the output to a readable format
predicted_labels = torch.argmax(output, dim=1)
print(classification_report(y_test, predicted_labels))

# check with a dummy
print(classification_report(y_test, # ground truth
                            [1 for x in y_test], # dummy which returns always a heart condition
                            zero_division = 0 # remove 'warn' default
                           )
     )

mms = MinMaxScaler()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(mms.fit_transform(X_train), y_train) # fitting on training
knn_pred = knn.predict(mms.fit_transform(X_val)) # checking on validation
print('Validation:\n', classification_report(y_val, knn_pred))

# Trying to tinkering with it further is meaningless considering the comparable performances
knn_pred = knn.predict(mms.fit_transform(X_test)) # checking on test
print('\nTest:\n', classification_report(y_test, knn_pred))
