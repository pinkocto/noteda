import itertools
import time
import os 
import pandas as pd
import datetime
import torch_geometric_temporal 
from .learners import * 
from .utils import *


class PLNR():
    def __init__(self,plans,loader,dataset_name=None,simulation_results=None):
        self.plans = plans
        # col = ['dataset', 'method', 'lags', 'nof_filters', 'epoch', 'mse','calculation_time']
        col = ['dataset', 'method', 'normal','lags', 'nof_filters', 'epoch', 'mse(train)','mse(test)','calculation_time']
        self.loader = loader
        self.dataset_name = dataset_name
        self.simulation_results = pd.DataFrame(columns=col) if simulation_results is None else simulation_results 
    def record(self,method,lags,nof_filters,epoch,mse_tr, mse_test, calculation_time):
        dct = {'dataset': self.dataset_name,
               'method': method, 
               'normal': 'X',
               'lags': lags,
               'nof_filters': nof_filters,
               'epoch': epoch,
               # 'mse': mse,
               'mse(train)': mse_tr,
               'mse(test)': mse_test,
               'calculation_time': calculation_time
              }
        simulation_result_new = pd.Series(dct).to_frame().transpose()
        self.simulation_results = pd.concat([self.simulation_results,simulation_result_new]).reset_index(drop=True)
    def save(self):
        if 'simulation_results' not in os.listdir(): 
            os.mkdir('simulation_results')
        fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.simulation_results.to_csv('./simulation_results/'+fname,index=False)    
        print("All results are stored in ./simulation_results/"+fname)
        
        
    
class PLNR_STGCN(PLNR):
    def simulate(self):
        for _ in range(self.plans['max_iteration']):  
            product_iterator = itertools.product(
                self.plans['method'], 
                self.plans['lags'], 
                self.plans['nof_filters'], 
                self.plans['epoch']
            )
            for prod_iter in product_iterator:
                method,lags,nof_filters,epoch = prod_iter
                self.dataset = self.loader.get_dataset(lags=lags)
                train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(self.dataset, train_ratio=0.7)
                lrnr = StgcnLearner(train_dataset,dataset_name=self.dataset_name)
                # if mrate > 0: 
                #     mtype = 'rand'
                #     mindex = rand_mindex(train_dataset,mrate=mrate)
                #     train_dataset = padding(train_dataset_miss = miss(train_dataset,mindex=mindex,mtype=mtype),interpolation_method=inter_method)
                # elif mrate ==0: 
                #     mtype = None
                #     inter_method = None 
                # if method == 'STGCN':
                #     lrnr = StgcnLearner(train_dataset,dataset_name=self.dataset_name)
                # elif method == 'IT-STGCN':
                #     lrnr = ITStgcnLearner(train_dataset,dataset_name=self.dataset_name)
                t1 = time.time()
                lrnr.learn(filters=nof_filters,epoch=epoch)
                t2 = time.time()
                evtor = Evaluator(lrnr,train_dataset,test_dataset)
                evtor.calculate_mse()
                # mse = evtor.mse['test']['total']
                mse_tr = evtor.mse['train']['total']
                mse_test = evtor.mse['test']['total']
                calculation_time = t2-t1
                # self.record(method,lags,nof_filters,epoch,mse,calculation_time)
                self.record(method, lags, nof_filters, epoch, mse_tr, mse_test, calculation_time)
            print('{}/{} is done'.format(_+1,self.plans['max_iteration']))
        self.save()


        
        
class PLNR2():
    def __init__(self,plans, data_dict ,dataset_name=None,simulation_results=None):
        self.plans = plans
        self.data_dict = data_dict
        col = ['dataset', 'method', 'normal','lags', 'nof_filters', 'epoch', 'mse(train)','mse(test)','calculation_time']
        self.dataset_name = dataset_name
        self.simulation_results = pd.DataFrame(columns=col) if simulation_results is None else simulation_results 
    def record(self,method,lags,nof_filters,epoch,mse_tr, mse_test, calculation_time):
        dct = {'dataset': self.dataset_name,
               'method': method, 
               'normal': 'O',
               'lags': lags,
               'nof_filters': nof_filters,
               'epoch': epoch,
               'mse(train)': mse_tr,
               'mse(test)': mse_test,
               'calculation_time': calculation_time
              }
        simulation_result_new = pd.Series(dct).to_frame().transpose()
        self.simulation_results = pd.concat([self.simulation_results,simulation_result_new]).reset_index(drop=True)
    def save(self):
        if 'simulation_results' not in os.listdir(): 
            os.mkdir('simulation_results')
        fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.simulation_results.to_csv('./simulation_results/'+fname,index=False)    
        print("All results are stored in ./simulation_results/"+fname)
        
        
class NORMAL_PLNR_STGCN(PLNR2):
    def simulate(self):
        for _ in range(self.plans['max_iteration']):  
            product_iterator = itertools.product(
                self.plans['method'], 
                self.plans['lags'], 
                self.plans['nof_filters'], 
                self.plans['epoch']
            )
            for prod_iter in product_iterator:
                method,lags,nof_filters,epoch = prod_iter
                """
                self.dataset = self.loader.get_dataset(lags=lags)
                train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(self.dataset, train_ratio=0.7)
                """
                data_dict = self.data_dict
                edges = np.array(data_dict["edges"]).T
                edge_weights = np.array(data_dict['weights'])
                scaled_edge_weights = minmaxscaler(edge_weights)
                edge_weight = scaled_edge_weights
                
                X_train = np.array(data_dict['FX'])[:1798]
                X_test = np.array(data_dict['FX'])[1798:]
                stacked_train = np.stack(X_train)
                stacked_test = np.stack(X_test)
                standardized_train = (stacked_train - np.mean(stacked_train, axis=0)) / (np.std(stacked_train, axis=0) + 10 ** -10)
                standardized_test = (stacked_test - np.mean(stacked_train, axis=0)) / (np.std(stacked_train, axis=0) + 10 ** -10)
                train_features = [standardized_train[i : i + lags, :].T for i in range(standardized_train.shape[0] - lags)]
                train_targets = [standardized_train[i + lags, :].T for i in range(standardized_train.shape[0] - lags)]
                test_features = [standardized_test[i : i + lags, :].T for i in range(standardized_test.shape[0] - lags)]
                test_targets = [standardized_test[i + lags, :].T for i in range(standardized_test.shape[0] - lags)]
                train_dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(edge_index= edges,
                                                                                            edge_weight = edge_weight,
                                                                                            features = train_features,
                                                                                            targets = train_targets)
                test_dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(edge_index= edges,
                                                                                            edge_weight = edge_weight,
                                                                                            features = test_features,
                                                                                            targets = test_targets)
                
                lrnr = NormalStgcnLearner(train_dataset,dataset_name=self.dataset_name)
                t1 = time.time()
                lrnr.learn(filters=nof_filters,epoch=epoch)
                t2 = time.time()
                evtor = Evaluator(lrnr,train_dataset,test_dataset)
                evtor.calculate_mse()
                # mse = evtor.mse['test']['total']
                mse_tr = evtor.mse['train']['total']
                mse_test = evtor.mse['test']['total']
                calculation_time = t2-t1
                self.record(method, lags, nof_filters, epoch, mse_tr, mse_test, calculation_time)
            print('{}/{} is done'.format(_+1,self.plans['max_iteration']))
        self.save()