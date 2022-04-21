# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 01:08:24 2022

@author: Akshatha V
"""
import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

df=pd.read_csv('C:\\Users\\Akshatha V\\Downloads\\PYTORCH_NOTEBOOKS\\PYTORCH_NOTEBOOKS\\Data\\iris.csv')
print(df.shape)

#METHOD 1 train test split using scikit learning
# =============================================================================
# features = df.drop('target', axis=1).values #axis is 1 as we want to remove the COLUMN
# label= df['target'].values
# X_train,X_test,y_train,y_test=train_test_split(features,label,test_size=0.2)
# # print(X_test.shape)
# # print(y_train.shape)
# X_train=torch.Tensor(X_train)
# X_test=torch.Tensor(X_test)
# #convert labels into long tensor and reshape to 2D
# Y_train=torch.LongTensor(y_train).reshape(-1,1)
# Y_test=torch.LongTensor(y_test).reshape(-1,1)
# =============================================================================

#METHOD2 - dataloader
data=df.drop('target',axis=1).values
labels=df['target'].values
iris= TensorDataset(torch.FloatTensor(data),torch.LongTensor(labels))

#creation of batches for dataloader
iris_loader = DataLoader(iris,batch_size=50,shuffle=True)

for i_batch,sample_batch in enumerate(iris_loader):
    print(i_batch, sample_batch)