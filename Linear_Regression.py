# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:11:14 2022

@author: Akshatha V
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn #for neural network functions

X= torch.linspace(1,50,50).reshape(-1,1) # 50 linearly spaced numbers from 1 to 50
# print(X)

#random error value creation
e = torch.randint(-8,9,(50,1),dtype=torch.float)
# print(e)

#setting up column matrix of y values
#adding error value to get a random variation otherwise the output will be fully linear
y=2*X+1+e
# print(y)

#to plot, convert X and y to numpy first
X=X.numpy()
y=y.numpy()

#plotting
# plt.scatter(X,y)
# plt.show()

#creation of linear model
# model=nn.Linear(1,1)
# print(model.weight, model.bias)

#setting up model class
class Model(nn.Module):
    def __init__(self,inp,outp):
        super().__init__()
        self.linear=nn.Linear(inp,outp)
    
    def forward(self,x):
        x=torch.tensor(x)
        y_pred = self.linear(x)
        return y_pred


model=Model(1,1)
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
# print(model.linear.weight, model.linear.bias)
y=torch.tensor(y)

epochs = 50
losses=[]

for i in range(epochs):

    y_pred=model.forward(X) #predicting forward pass
    loss=criterion(y_pred,y)#calculate our loss
    losses.append(loss.item())#recording the loss
    # print(f"epoch{i} loss:{loss.item()} weight:{model.linear.weight.item()} bias:{model.linear.bias.item()}")
    
    optimizer.zero_grad() #resetting stored gradient for each epoch
 
    loss.backward() #backpropagation
  
    optimizer.step()#updates the hyperparameters of the model
 
# plt.plot(range(epochs),losses)

x=np.linspace(0.0,50.0,50)
# print(x)
current_weight=model.linear.weight.item()
current_bias=model.linear.bias.item()
predicted_y= current_weight*x + current_bias
# print(predicted_y)

plt.scatter(X,y.numpy())
plt.plot(x,predicted_y,'b')
#iteration over all the model parameters
# for name, param in model.named_parameters():
#     print(name,'\t', param.item())