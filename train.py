import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def transform_regularizer(trans):
    k = trans.shape[1]

    identity = torch.eye(k).to(trans.device)
    trans_square = torch.bmm(trans.transpose(1, 2), trans)
    reg_loss = torch.mean(torch.norm(identity - trans_square, dim=(1, 2)))
    return reg_loss

def total_loss(l,trans,trans_feat):
    return l + 0.001*(transform_regularizer(trans)+transform_regularizer(trans_feat))

class TNet(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,64,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,128,1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024,1)

        self.mlp = nn.Sequential(nn.Linear(1024,512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Linear(512,256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.linear1 = nn.Linear(256,k*k)

        nn.init.constant_(self.linear1.weight,0)
        identity = torch.eye(k).reshape(-1)
        self.linear1.bias.data.copy_(identity.view(-1))

                                
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

       
        #x = self.maxpool(x)
        x = torch.max(x,dim=2)[0]
        #x = x.reshape(x.shape[0],-1)

        x = self.mlp(x)

        x = self.linear1(x)
        x = x.reshape(x.shape[0],self.k,self.k)
        return x

        
class PointNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.T1 = TNet(3)

        self.mlp1 = nn.Sequential(nn.Conv1d(3,64,1),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Conv1d(64,64,1),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
        )

        self.T2 = TNet(64)

        self.mlp2 = nn.Sequential(nn.Conv1d(64,64,1),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Conv1d(64,128,1),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Conv1d(128,1024,1),
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU())
        
        self.maxpool = nn.MaxPool1d(1024,1)

        self.mlp3 = nn.Sequential(nn.Linear(1024,512),
                                  nn.BatchNorm1d(512),
                                  nn.Dropout(0.3),
                                  nn.ReLU(),
                                  nn.Linear(512,256),
                                  nn.BatchNorm1d(256),
                                  nn.Dropout(0.3),
                                  nn.ReLU(),
                                  nn.Linear(256,10))
    def forward(self,x):
        T1 = self.T1.forward(x)
        x = torch.bmm(T1,x)

        x = self.mlp1(x)

        T2 = self.T2.forward(x)
        x = torch.bmm(T2,x)

        x = self.mlp2(x)      

    

        #x = self.maxpool(x)
        x = torch.max(x,dim=2)[0]
        #x = x.reshape(x.shape[0],-1)
        #print(x.shape)
        #x = x.transpose(1,2)

        x = self.mlp3(x)
        return x ,T1,T2 

def evluate_accuracy_gpu(net,test_iter):
    net.eval()
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for x,y in test_iter:
            x,y = x.to(device),y.to(device)
       
            y_hat,_,_ = net(x)
            l = loss(y_hat,y)
            
            metric.add(l*x.shape[0],d2l.accuracy(y_hat,y),x.shape[0])
    return metric[0]/metric[2],metric[1]/metric[2]
            
'''
X = torch.randn(size=(5,3,1024))  #5个数据，每个500个点
net = PointNet()

y = net(X)
print(y.shape)
'''


epochs,pointsnum,batch_size,lr,weight_decay = 10,1024,32,0.001,0.0


device = torch.device('cuda:0')
net = PointNet()
net.to(device)




X,y_labels = torch.load("train_features.pt"),torch.load("train_labels.pt")
y_labels = d2l.argmax(y_labels,axis=1)

X_test,y_labels_test = torch.load("test_features.pt"),torch.load("test_labels.pt")
y_labels_test = d2l.argmax(y_labels_test,axis = 1)



train_iter = load_array((X,y_labels),batch_size,True)
test_iter = load_array((X_test,y_labels_test),batch_size,False)

loss = nn.CrossEntropyLoss()
#update = torch.optim.SGD(net.parameters(),lr=lr)
update = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)


train_errs,test_errs = [],[]

for epoch in range(epochs):
    metric = d2l.Accumulator(3)
    net.train()
    for x,y in train_iter:
        #print(y) 
        update.zero_grad()
        x,y = x.to(device),y.to(device)
       
        y_hat,t1,t2 = net(x)
        l = loss(y_hat,y)
        l = total_loss(l,t1,t2)
        l.backward()
        update.step()
    
        with torch.no_grad():
            metric.add(l * x.shape[0], d2l.accuracy(y_hat,y), x.shape[0])
            #print(metric[1])
    train_l = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]

    test_l,test_acc = evluate_accuracy_gpu(net,test_iter)


    train_errs.append(train_l)
    test_errs.append(test_l)

    print(train_acc,test_acc)


x = range(len(train_errs))
plt.plot(x,test_errs,color="blue",linestyle="-")
plt.plot(x,train_errs,color ="red",linestyle = "-.")
plt.yscale('log')
plt.legend(['test','train'])
plt.show()