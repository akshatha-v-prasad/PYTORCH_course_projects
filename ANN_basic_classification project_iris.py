# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:24:06 2022

@author: Akshatha V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    
    def __init__(self, in_features=4, hidden1=8,hidden2=9,out_labels=3):
        #how many layers?
        #input layer(4 features) to h1 (N) to h1(N) to output(3 classes)
        super().__init__() #instantiates the inherited class which is nn.Module
        self.fc1 = nn.Linear(in_features,hidden1)
        self.fc2= nn.Linear(hidden1,hidden2)
        self.fc3= nn.Linear(hidden2, out_labels)
        
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
model=Model()

#read data
df=pd.read_csv('C:\\Users\\Akshatha V\\Downloads\\PYTORCH_NOTEBOOKS\\PYTORCH_NOTEBOOKS\\Data\\iris.csv')
print(df.head())

#splitting of data
features = df.drop('target',axis=1).values #.values is to convert to numpy arrays
label =df['target'].values
print(label)
print(features)
x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.25)

#coverting to tensor
x_train = torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
#no need to reshape output to 2D as we are using CrossEntropy
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

criterion=nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(),lr=0.01)

print(model.parameters)

#TRAINING
#epoch=one run through all the training data
epochs=100
train_loss=[]

for i in range(epochs):
    optimizer.zero_grad()
    
    #going forward in the model to get a prediction
    y_pred=model.forward(x_train)
    
    #calculation of the loss to see how the model is doing
    loss=criterion(y_pred,y_train)
    train_loss.append(loss.item())
    #printing loss for every 10 epochs
    if i%10==0:
        print(f'Epoch {i} and loss is: {loss}')
    #backpropagation
    loss.backward()
    optimizer.step()  

print(train_loss)

plt.plot(range(epochs),train_loss)

#VALIDATION
with torch.no_grad(): #no backpropagation
    y_eval = model.forward(x_test)
    vali_loss = criterion(y_eval,y_test)

print(vali_loss)

#how many did we classify correctly

correct = 0

with torch.no_grad():
    for i,data in enumerate(x_test):
        y_val=model.forward(data)
        print(y_val.argmax(), y_test[i])
        if y_val.argmax().item()== y_test[i]:
            correct+=1
print(i+1)
print(correct) 

#for SAVING the model

# =============================================================================
# torch.save(model.state_dict(),"C:\\Users\\Akshatha V\\Udemy_Pytorch\\my_iris_model_one.pt")
# # torch.save(model,"C:\\Users\\Akshatha V\\Udemy_Pytorch\\my_iris_model_one.pt")   
# #how to load the saved model
# new_model=Model()
# new_model.load_state_dict(torch.load('my_iris_model_one.pt'))
# print(new_model.eval())
# =============================================================================
    
#To predict an entire new data
unknown_iris_datapoint = torch.tensor([5.6,3.7,2.2,0.5]) #it should be class 0

with torch.no_grad():
    print(model(unknown_iris_datapoint))
    print(model(unknown_iris_datapoint).argmax())
    