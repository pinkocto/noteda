import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F
#import torch_geometric_temporal
from torch_geometric_temporal.nn.recurrent import GConvGRU

# utils
import copy
#import time
#import pickle
#import itertools
#from tqdm import tqdm
#import warnings

# rpy2
#import rpy2
#import rpy2.robjects as ro 
# from rpy2.robjects.vectors import FloatVector
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# import rpy2.robjects.numpy2ri as rpyn
#igraph = importr('igraph') # import igraph 


# from .utils import convert_train_dataset
from .utils import DatasetLoader



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class StgcnLearner:
    def __init__(self,train_dataset,dataset_name = None):
        self.train_dataset = train_dataset
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.dataset_name = str(train_dataset) if dataset_name is None else dataset_name
        self.method = 'STGCN'
    def learn(self,filters=32,epoch=50):
        self.model = RecurrentGCN(node_features=self.lags, filters=filters)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for e in range(epoch):
            for t, snapshot in enumerate(self.train_dataset):
                yt_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                cost = torch.mean((yt_hat-snapshot.y)**2)
                cost.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # print('{}/{}'.format(e+1,epoch),end='\r')
            print('{}/{}'.format(e,epoch),end='\r')
        # recording HP
        self.nof_filters = filters
        self.epochs = epoch+1
    def __call__(self,dataset):
        X = torch.tensor(dataset.features).float()
        y = torch.tensor(dataset.targets).float()
        yhat = torch.stack([self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr) for snapshot in dataset]).detach().squeeze().float()
        return {'X':X, 'y':y, 'yhat':yhat} 
    
        
# class EPTStgcnLearner(StgcnLearner):
#     def __init__(self,train_dataset,dataset_name = None):
#         super().__init__(train_dataset)
#         self.method = 'EPT-STGCN'
#     def learn(self,filters=32,epoch=50):
#         self.model = RecurrentGCN(node_features=self.lags, filters=filters)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
#         self.model.train()
#         train_dataset_temp = copy.copy(self.train_dataset)
#         for e in range(epoch):
#             f,lags = convert_train_dataset(train_dataset_temp)
#             T,N = f.shape 
#             data_dict_temp = {
#                 'edges':self.train_dataset.edge_index.T.tolist(), 
#                 'node_ids':{'node'+str(i):i for i in range(N)}, 
#                 'FX':f
#             }
#             train_dataset_temp = DatasetLoader(data_dict_temp).get_dataset(lags=self.lags)  
#             for t, snapshot in enumerate(train_dataset_temp):
#                 yt_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#                 cost = torch.mean((yt_hat-snapshot.y)**2)
#                 cost.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#             print('{}/{}'.format(e+1,epoch),end='\r')
#         # record
#         self.nof_filters = filters
#         self.epochs = epoch+1