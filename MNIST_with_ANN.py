# -*- coding: utf-8 -*-
"""
Created on Sun May  1 18:32:00 2022

@author: Akshatha V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

#torchvision library used to view image dataset
#main aim is to convert the images into tensors using TRANSFORM
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='C:\\Users\\Akshatha V\\Downloads\\PYTORCH_NOTEBOOKS\\PYTORCH_NOTEBOOKS\\Data', train =True, download=True, transform=transform)
test_data = datasets.MNIST(root='C:\\Users\\Akshatha V\\Downloads\\PYTORCH_NOTEBOOKS\\PYTORCH_NOTEBOOKS\\Data', train =False, download=True, transform=transform)

print(train_data[0])
#the train data is in the form of a tuple --> ((28,28),label)
#grayscale image but its shown in this purple and yellow default to make it easier for colorblind people to see

# image,label= train_data[0]
# plt.imshow(image.reshape((28,28)))

#batching
train_loader=DataLoader(train_data, batch_size=100,shuffle=True)
test_loader=DataLoader(test_data, batch_size=500,shuffle=False)

for images, labels in train_loader:
    break


#model
class MLP(nn.Module):
    def __init__(self, in_size=784, out_size=10, layers=[120,84]):
        super().__init__()
        self.fc1=nn.Linear(in_size,layers[0])
        self.fc2=nn.Linear(layers[0],layers[1])
        self.fc3=nn.Linear(layers[1],out_size)
    
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x) #as this is a multiclass classification, we dont pass the last layer into relu. we use softmax
        
        return F.log_softmax(x, dim=1)

model=MLP()
print(model)
criterion= nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

#reshaping the input
images.view(-1,784)

#training
import time
start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        
        # Apply the model
        y_pred = model(X_train.view(100, -1))  # Here we flatten X_train
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print interim results
        if b%200 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{100*b:6}/60000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(100*b):7.3f}%')
    
    # Update train loss & accuracy for the epoch
    train_losses.append(loss.item())
    train_correct.append(trn_corr)
        
    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test.view(500, -1))  # Here we flatten X_test

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()
    
    # Update test loss & accuracy for the epoch
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)
        
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend();

# Extract the data all at once, not in batches
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test), -1))  # pass in a flattened view of X_test
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')
