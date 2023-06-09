# Importing the necessary libraries and packages
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import keras
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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

# Describe the data set after transformation/cleansing
print(PhysInfo_clean.shape)
PhysInfo_clean.describe().T
print(PhysInfo_clean.dtypes)
  # Plot histograms for each variable
PhysInfo_clean.hist(figsize = (12, 12))
plt.show()
  # Create a heatmap of the data set
plt.figure(figsize = (12, 12))
sns.heatmap(PhysInfo_clean.corr(), annot=True, fmt=' .1f') #method1
plt.show()
sns.heatmap(PhysInfo_clean.isnull(),cmap = 'magma',cbar = False); #method2
  # Plot the means of each applicable feature of the data set between those with heart disease and those without
with = PhysInfo_clean[PhysInfo_clean['HeartDisease'] == 1].describe().T
without = PhysInfo_clean[PhysInfo_clean['HeartDisease'] == 0].describe().T
colors = ['#13D182', '#C413D1']
fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (5,5))
plt.subplot(1,2,1)
sns.heatmap(with[['mean']],annot = True,cmap = colors,linewidths = 0.5,linecolor = 'black',cbar = False, fmt = '.2f',)
plt.title('Heart Disease');
plt.subplot(1,2,2)
sns.heatmap(without[['mean']],annot = True,cmap = colors,linewidths = 0.5,linecolor = 'black',cbar = False, fmt = '.2f')
plt.title('No Heart Disease');
fig.tight_layout(pad = 2)

# Division of numerical and categorical variables
columns = list(PhysInfo_clean.columns)
categorical = []
numerical = []
for i in columns:
  if len(PhysInfo_clean[i].unique()) > 6:
    numerical.append(i)
  else:
    categorical.append(i)
print('Categorical Features:',*categorical)
print('Numerical Features:',*numerical)

# Create a deep copy of Categorical Variables, converting them into numerical values for visualization & modeling purposes
label = LabelEncoder()
deep = PhysInfo_clean.copy(deep = True)
deep['Sex'] = label.fit_transform(deep['Sex'])
deep['ChestPainType'] = label.fit_transform(deep['ChestPainType'])
deep['RestingECG'] = label.fit_transform(deep['RestingECG'])
deep['ExerciseAngina'] = label.fit_transform(deep['ExerciseAngina'])
deep['ST_Slope'] = label.fit_transform(deep['ST_Slope'])

# Find and Plot the Distribution of the Variables
  # Categorical Distribution
fig, ax =  plt.subplots(nrows = 3, ncols = 2, figsize = (10, 15))
for i in range(len(categorical) - 1):
  plt.subplot(3, 2, i+1)
  sns.distplot(deep[categorical[i]], kde_kws = {'bw' : 1}, color = colors[0]);
  title = 'Histogram of ' + categorical[i]
  plt.title(title)
plt.figure(figsize = (5, 5))
sns.distplot(deep[categorical[len(categorical) - 1]], kde_kws = {'bw' : 1}, color = colors[0])
title = 'Histogram of ' + categorical[len(categorical) - 1]
plt.title(title);
  # Numerical Distribution
fig, ax =  plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))
for i in range(len(numerical) - 1):
  plt.subplot(2, 2, i+1)
  sns.distplot(PhysInfo_clean[numerical[i]], color = colors[0]);
  title = 'Histogram of ' + numerical[i]
  plt.title(title)
plt.show()
plt.figure(figsize = (5, 5))
sns.distplot(deep[numerical[len(numerical) - 1]], kde_kws = {'bw' : 1}, color = colors[0])
title = 'Histogram of ' + numerical[len(numerical) - 1]
plt.title(title);

# Determine the balance of the data set (w.r.t. Heart Disease v. No Heart Disease)
HeartDisease = list(data['HeartDisease'].value_counts())
pie = [HeartDisease[1] / sum(HeartDisease) * 100, HeartDisease[0] / sum(HeartDisease) * 100]
plt.figure(figsize = (5, 5))
plt.pie(pie, labels = ['No Heart Disease', 'Heart Disease'', autopct = '%1.1f%%', startangle = 90, explode = (0.1, 0), colors = colors, wedgeprops = {'edgecolor' : 'black', 'linewidth' : 1, 'antialiased' : True})
plt.title('Heart Disease %');
plt.show()

# Categorical Variables w.r.t. Heart Disease

# Numerical Variable w.r.t. Heart Disease
