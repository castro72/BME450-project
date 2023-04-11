# Importing the necessary libraries and packages
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import keras
import sklearn
import random
from micrograd.engine import Value
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')

# Importing and reading the relevant data set
PhysInfo = pd.read_csv(r'/kaggle/input/heart-failure-prediction/heart.csv')
# Print the head (first five rows) of the data set
PhysInfo.head()

# Analyze and print the shape of the data set
print( 'Shape of PhysInfo: {}'.format(PhysInfo.shape))
print(PhysInfo.columns)
PhysInfo.info()
#print (PhysInfo.loc[1]) (?)

# Check for null values present in the data set
sns.heatmap(PhysInfo.isnull(), cmap = 'magma', cbar = False);

# Remove any missing data or null values from the data
PhysInfo_clean = PhysInfo[~PhysInfo.isin(['?'])]
PhysInfo_clean = PhysInfo_clean.dropna(axis=0)

# Convert the data into features that the machine will understand
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

PhysInfo_clean['Oldpeak'] = mms.fit_transform(PhysInfo_clean[['Oldpeak']]) # Oldpeak is normalized because it has a right skewed distribution
PhysInfo_clean['Age'] = ss.fit_transform(PhysInfo_clean[['Age']]) # Age and all following variables are scaled down (standardized) because they are normally distributed
PhysInfo_clean['RestingBP'] = ss.fit_transform(PhysInfo_clean[['RestingBP']])
PhysInfo_clean['Cholesterol'] = ss.fit_transform(PhysInfo_clean[['Cholesterol']])
PhysInfo_clean['MaxHR'] = ss.fit_transform(PhysInfo_clean[['MaxHR']])
PhysInfo_clean.head()

# Split the data into patient data and target (Heart Disease vs. No Heart Disease)
X = PhysInfo_clean.drop('HeartDisease', axis = 1)
y = PhysInfo_clean['HeartDisease']

# Create Training and Testing Datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)

# Micrograd Neural Network
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# Define the Multilayer Perceptron
P = MLP(11, [918, 1]) ## Maybe change this from 918 to something smaller?

# Writing out the training loop

training_cycles = 20
learning_rate = 0.01

for i in range(training_cycles):

    # forward pass
    y_predicted = [P(x) for x in X_train]

    # recalculate loss
    loss = sum([(y_output - y_ground_truth)**2 for y_ground_truth, y_output in zip(y_train, y_predicted)])

    # backward pass
    
    # COMMON BUG: Forgetting to zero-out your gradients before
    # running the next backward pass
    
    for p in P.parameters():
        p.grad = 0
    
    loss.backward()

    # gradient updates
    for p in P.parameters():
        p.data += -learning_rate * p.grad
        
    # print current loss
    print(f'i={i}: loss={loss}')
