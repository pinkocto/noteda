import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import animation
import json
import urllib

# # torch
import torch
# import torch.nn.functional as F
import torch_geometric_temporal
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
# from torch_geometric_temporal.nn.recurrent import GConvGRU


# utils
#import copy
#import time
import pickle
import itertools
#from tqdm import tqdm
#import warnings


temporal_signal_split = torch_geometric_temporal.signal.temporal_signal_split

def minmaxscaler(arr):
    arr = arr - arr.min()
    arr = arr/arr.max()
    return arr 


class DatasetLoader(object):
    """Hourly solar radiation of observatories from South Korean  for 2 years. 
    Vertices represent 44 cities and the weighted edges represent the strength of the relationship. 
    The target variable allows regression operations. 
    (The weight is the correlation coefficient of solar radiation by region.)
    """

    def __init__(self, url):
        self.url = url
        self._read_web_data()
        
    def _read_web_data(self):
        self._dataset = json.loads(urllib.request.urlopen(self.url).read().decode())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        # self._edge_weights = np.array(self._dataset["weights"]).T
        edge_weights = np.array(self._dataset["weights"]).T
        scaled_edge_weights = minmaxscaler(edge_weights)
        self._edge_weights = scaled_edge_weights

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["FX"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    

    
class Evaluator:
    def __init__(self,learner,train_dataset,test_dataset):
        self.learner = learner
        # self.learner.model.eval()
        try:self.learner.model.eval()
        except:pass
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lags = self.learner.lags
        rslt_tr = self.learner(self.train_dataset) 
        rslt_test = self.learner(self.test_dataset)
        self.X_tr = rslt_tr['X']
        self.y_tr = rslt_tr['y']
        self.f_tr = torch.concat([self.train_dataset[0].x.T,self.y_tr],axis=0).float()
        self.yhat_tr = rslt_tr['yhat']
        self.fhat_tr = torch.concat([self.train_dataset[0].x.T,self.yhat_tr],axis=0).float()
        self.X_test = rslt_test['X']
        self.y_test = rslt_test['y']
        self.f_test = self.y_test 
        self.yhat_test = rslt_test['yhat']
        self.fhat_test = self.yhat_test
        self.f = torch.concat([self.f_tr,self.f_test],axis=0)
        self.fhat = torch.concat([self.fhat_tr,self.fhat_test],axis=0)
    def calculate_mse(self):
        test_base_mse_eachnode = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean(axis=0).tolist()
        test_base_mse_total = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean().item()
        train_mse_eachnode = ((self.y_tr-self.yhat_tr)**2).mean(axis=0).tolist()
        train_mse_total = ((self.y_tr-self.yhat_tr)**2).mean().item()
        test_mse_eachnode = ((self.y_test-self.yhat_test)**2).mean(axis=0).tolist()
        test_mse_total = ((self.y_test-self.yhat_test)**2).mean().item()
        self.mse = {'train': {'each_node': train_mse_eachnode, 'total': train_mse_total},
                    'test': {'each_node': test_mse_eachnode, 'total': test_mse_total},
                    'test(base)': {'each_node': test_base_mse_eachnode, 'total': test_base_mse_total},
                   }
    def _plot(self,*args,t=None,h=2.5,max_node=5,**kwargs):
        T,N = self.f.shape
        if t is None: t = range(T)
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            ax[n].plot(t,self.f[:,n],color='gray',*args,**kwargs)
            ax[n].set_title('node='+str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    def plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _mse2= self.mse['test']['each_node'][i]
            _mse3= self.mse['test(base)']['each_node'][i]
            _mrate = self.learner.mrate_eachnode if set(dir(self.learner.mrate_eachnode)) & {'__getitem__'} == set() else self.learner.mrate_eachnode[i]
            _title = 'node{0}, mrate: {1:.2f}% \n mse(train) = {2:.2f}, mse(test) = {3:.2f}, mse(test_base) = {4:.2f}'.format(i,_mrate*100,_mse1,_mse2,_mse3)
            a.set_title(_title)
            _t1 = self.lags
            _t2 = self.yhat_tr.shape[0]+self.lags
            _t3 = len(self.f)
            a.plot(range(_t1,_t2),self.yhat_tr[:,i],label='fitted (train)',color='C0')
            a.plot(range(_t2,_t3),self.yhat_test[:,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n mrate: {2:.2f}% \n interpolation:{3} \n epochs={4} \n number of filters={5} \n lags = {6} \n mse(train) = {7:.2f}, mse(test) = {8:.2f}, mse(test_base) = {9:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.mrate_total*100,self.learner.interpolation_method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.tight_layout()
        return fig