#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:34:13 2023

@author: jakecastro & Toby Miller
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

#Create Training, Validation, and Test sets
n1 = int(0.8 * int(len(df)))
n2 = int(0.9 * int(len(df)))

df_tr = df_in.iloc[:n1]
df_val = df_in.iloc[n1:n2]
df_tst = df_in.iloc[n2:]

print(df_tr)
print(df_val)
print(df_tst)



# print(df.iloc[:1])


