#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:34:13 2023

@author: jakecastro
"""
import torch
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')

## We want to import table
df = pd.read_csv('/Users/jakecastro/Desktop/Classes/BME450/heart.csv')

print(df.head())

# Separate Results from inputs

df_in = df.iloc[: , :-1] # data table without "Heart Disease" Column, so all the inputs

df_results = df[["HeartDisease"]] # output [0] or [1] for Heart Disease

print(df_in)
print(df_results)

##Dividing Table into Quantitative and Categorical Metrics

col = list(df_in.columns)
print("\nColumn Names:", col, "\n")
categorical_features = []
numerical_features = []
for i in col:
    if len(df_in[i].unique()) > 6: ## if metric has more than 6 unique values it is numerical
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('\nCategorical Columns :',*categorical_features, sep= " , ")
print('\nNumerical Columns :',*numerical_features, sep= " , ")

## Converting Categorical Columns into Numerical

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_enc = df_in.copy(deep = True)

df_enc['Sex'] = le.fit_transform(df_in['Sex'])
df_enc['ChestPainType'] = le.fit_transform(df_in['ChestPainType'])
df_enc['RestingECG'] = le.fit_transform(df_in['RestingECG'])
df_enc['ExerciseAngina'] = le.fit_transform(df_in['ExerciseAngina'])
df_enc['ST_Slope'] = le.fit_transform(df_in['ST_Slope'])

print("\nConversion of Categorical Values to Numerical Values\n")

for i in range(len(categorical_features)):
    print("\n",categorical_features[i],":", df_in[categorical_features[i]].unique(), "-->", df_enc[categorical_features[i]].unique())
print("\n")
#Create Training, Validation, and Test sets
n1 = int(0.8 * int(len(df)))
n2 = int(0.9 * int(len(df)))

#inputs of each row

df_tr = df_enc.iloc[:n1]
df_val = df_enc.iloc[n1:n2]
df_tst = df_enc.iloc[n2:]

#expected outcome
ys_tr = df_results.iloc[:n1]
ys_val = df_results.iloc[n1:n2]
ys_tst = df_results.iloc[n2:]

print("\nTraining Dataset Inputs: \n",df_tr)
print("\n")
print("\nValidation Dataset Inputs:\n", df_val)
print("\n")
print("\nTest Dataset Inputs:\n" , df_tst)
print("\n")


print("\nTraining Dataset Expected Outputs: \n" ,ys_tr)
print("\n")
print("Validation Dataset Expected Outputs: \n" ,ys_val)
print("\n")
print("Test Dataset Expected Outputs: \n" , ys_tst)


##Training 

xs = [[]] ## each row of 11 inputs
for i in range(len(df_tr)):
    xs.append((df_tr.iloc[i, 0], df_tr.iloc[i, 1], df_tr.iloc[i, 2], df_tr.iloc[i, 3], 
          df_tr.iloc[i, 4], df_tr.iloc[i, 5], df_tr.iloc[i, 6], df_tr.iloc[i, 7], 
          df_tr.iloc[i, 8], df_tr.iloc[i, 9], df_tr.iloc[i, 10]))
del(xs[0])
    
# print("\n")
# print(len(xs), len(xs[0]))
# print(xs)

ytr_pred = [0] * len(df_tr.index)
# print(ytr_pred)


##Validation

xs = [[]] ## each row of 11 inputs
for i in range(len(df_val)):
    xs.append((df_val.iloc[i, 0], df_val.iloc[i, 1], df_val.iloc[i, 2], df_val.iloc[i, 3], 
          df_val.iloc[i, 4], df_val.iloc[i, 5], df_val.iloc[i, 6], df_val.iloc[i, 7], 
          df_val.iloc[i, 8], df_val.iloc[i, 9], df_val.iloc[i, 10]))
del(xs[0])
    
# print("\n")
# print(len(xs), len(xs[0]))
# print(xs)

yval_pred = [0] * len(df_val.index)
# print(ytr_pred)

#Testing

xs = [[]] ## each row of 11 inputs
for i in range(len(df_tst)):
    xs.append((df_tst.iloc[i, 0], df_tst.iloc[i, 1], df_tst.iloc[i, 2], df_tst.iloc[i, 3], 
          df_tst.iloc[i, 4], df_tst.iloc[i, 5], df_tst.iloc[i, 6], df_tst.iloc[i, 7], 
          df_tst.iloc[i, 8], df_tst.iloc[i, 9], df_tst.iloc[i, 10]))
del(xs[0])
    
# print("\n")
# print(len(xs), len(xs[0]))
# print(xs)

ytst_pred = [0] * len(df_tst.index)
# print(ytr_pred)
