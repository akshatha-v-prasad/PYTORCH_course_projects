# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:40:47 2022

@author: Akshatha V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time

df= pd.read_csv("C:\\Users\\Akshatha V\\Downloads\\PYTORCH_NOTEBOOKS\\PYTORCH_NOTEBOOKS\\Data\\NYCTaxiFares.csv")
# print(df.head())

#Regression problem so the target is fare_amount

#-----------------FEATURE ENGINEERING-------------------------
# direct longitude and latitude will not be of much help.
#Haversine formula = calculation of distance using latitude and logintude

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

df['distance']= haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


#conversion of pickup date time to a date-time object(which is currently a string)
df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])
print(df.head())

#time difference between NYC time and UTC -> 4hrs difference
df['EDTdate']= df['pickup_datetime']- pd.Timedelta(hours=4)
df['hours']=df['EDTdate'].dt.hour
df['AMPM']=np.where(df['hours']<12,'AM',"PM")
df['__Day'] = df['EDTdate'].dt.strftime("%a")
print(df.head())
 
#now, AMPM and __day columns have become categorical..so have to separate continuous and categorical data
#-----------------------CATEGORICAL and CONTINUOUS features------------

category_cols =['hours','AMPM','__Day']
conti_cols=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','passenger_count','distance']

label_col=['fare_amount']

#Change the type of the categorical cols to category

for i in category_cols:
    df[i]=df[i].astype('category')

#conversion of all the categorical cols into numpy arrays
hour = df['hours'].cat.codes.values
amorpm=df['AMPM'].cat.codes.values
days =df['__Day'].cat.codes.values

#stacking them together
categories = np.stack([hour,amorpm,days],axis=1)
print(categories)

#OR just do:

continuous = np.stack([df[i].values for i in conti_cols], 1)

#conversion to tensors
categories=torch.tensor(categories,dtype=torch.int64)
continuous=torch.tensor(continuous,dtype=torch.float)
labels=torch.tensor(df[label_col].values,dtype=torch.float) #shape should be 2D

print(categories.shape,continuous.shape,labels.shape)

#embedding
category_size = [len(df[i].cat.categories) for i in category_cols]
embedded_size = [(size, min(50, (size+1)//2)) for size in category_size]
print(embedded_size)

class TabularModel(nn.Module):
    def __init__(self,embedded_size, n_cont, out_size, layers, prob=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in embedded_size])
        self.emb_drop = nn.Dropout(prob)
        self.bn_cont = nn.BatchNorm1d(n_cont)#normalization of the continuous data
        
        layerlist = []
        n_emb = sum((nf for ni,nf in embedded_size))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(prob))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_size))
            
        self.layers = nn.Sequential(*layerlist)
      
    def forward(self,x_cat, x_cont):
       embeddings = []
       for i,e in enumerate(self.embeds):
           embeddings.append(e(x_cat[:,i]))
       x = torch.cat(embeddings, 1)
       x = self.emb_drop(x)
       
       x_cont = self.bn_cont(x_cont)
       x = torch.cat([x, x_cont], 1)
       x = self.layers(x)
       return x
       
model = TabularModel(embedded_size, continuous.shape[1], 1, [200,100], prob=0.4)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#TRAIN-TEST split

batch_size = 60000
test_size = int(batch_size * .2)

cat_train = categories[:batch_size-test_size]
cat_test = categories[batch_size-test_size:batch_size]
con_train = continuous[:batch_size-test_size]
con_test = continuous[batch_size-test_size:batch_size]
y_train = labels[:batch_size-test_size]
y_test = labels[batch_size-test_size:batch_size]

#TRAINING MODEL
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss.item())
    
    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
plt.plot(range(epochs), losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch');

#VALIDATION
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(f'RMSE: {loss:.8f}')

#PREDICTION
print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
for i in range(50):
    diff = np.abs(y_val[i].item()-y_test[i].item())
    print(f'{i+1:2}. {y_val[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')