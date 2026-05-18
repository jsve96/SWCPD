import numpy as np
import pandas as pd
import os

import ruptures
import ruptures.metrics
from joblib import Parallel, delayed


def get_d_score(x):
    d_score = []
    for i in range(2, len(x)):
        i_d = np.abs((x[i]-x[i-1]) / (x[i-1] - x[i-2] + 10**-6))
        if i_d == 0:
            i_d = 2
        d_score.append(i_d)
    d_score = [d_score[0]] + d_score + [d_score[-1]]
    return np.array(d_score)




class RupturesBinseg(object):
    
    def __init__(self, model='rbf', jump=10):
        self.model = model
        self.jump = jump
    
    def run_predictions(self, X, n_bkps):
        algo = ruptures.Binseg(model=self.model, min_size=2, jump=self.jump).fit(X)
        my_bkps = algo.predict(n_bkps=n_bkps)
        return my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.n_bkps = np.arange(1, 41, 1)
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.reco_bkps = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.n_bkps)
        self.reco_bkps = np.asarray(self.reco_bkps,dtype="object")
        
        # compute gain for the found bkps
        cost = ruptures.costs.CostRbf().fit(X)
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        
        # gen score
        score = np.zeros(len(X))
        score[best_bkps[:-1]] = 1
        
        return score, best_bkps
    
    

class RupturesPelt(object):
    
    def __init__(self, model='rbf', jump=10):
        self.model = model
        self.jump = jump
    
    def run_predictions(self, X, pen):
        algo = ruptures.Pelt(model=self.model, min_size=2, jump=self.jump).fit(X)
        my_bkps = algo.predict(pen=pen)
        return my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.penalties = np.linspace(0, 10, 20)
        self.gains = []
        self.reco_bkps = []
        self.n_bkps = []
        
        # run Pelt models with different n_bkps
        self.reco_bkps = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.penalties)
        self.reco_bkps = np.asarray(self.reco_bkps,dtype='object')
        self.n_bkps = np.array([len(i) for i in self.reco_bkps])
        #print("self.n_bkps: ", self.n_bkps)
        
        # compute gain for the found bkps
        cost = ruptures.costs.CostRbf().fit(X)
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
                
        # gen score
        score = np.zeros(len(X))
        score[best_bkps[:-1]] = 1

        return score, best_bkps
    
    
    
class RupturesWindow(object):
    
    def __init__(self, model='rbf', jump=10, width=100):
        self.model = model
        self.jump = jump
        self.width = width
    
    def run_predictions(self, X, n_bkps):
        algo = ruptures.Window(model=self.model, width=self.width, min_size=2, jump=self.jump).fit(X)
        my_bkps = algo.predict(n_bkps=n_bkps)
        return my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.n_bkps = np.arange(1, 41, 1)
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.reco_bkps = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.n_bkps)
        self.reco_bkps = np.asarray(self.reco_bkps, dtype='object')
        
        # compute gain for the found bkps
        cost = ruptures.costs.CostRbf().fit(X)
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        
        # gen score
        score = np.zeros(len(X))
        score[best_bkps[:-1]] = 1
        
        return score, best_bkps
    
    
    
    
from online_change_clf import ChangePointDetectionOnline 
import itertools


class OnlineCLF(object):
    
    def __init__(self, lag_size=100, height=None, smooth=True):
        
        self.lag_size = lag_size
        self.height = height
        self.smooth = smooth
        
        self.params = list(itertools.product([1, 10], [1, 10], [1, 10], [0.01, 0.1], [1]))
    
    def run_predictions(self, X, params):
        algo = ChangePointDetectionOnline(net='auto', scaler='auto', metric="KL_sym", lag_size=self.lag_size, 
                                         batch_size=params[0], step=params[1], n_epochs=params[2], lr=params[3], 
                                          periods=params[4] , lam=0.0, optimizer='Adam')
        # Detect change points
        score, my_bkps = algo.predict(X, height=self.height, smooth=self.smooth)
        if len(my_bkps) == 0:
            my_bkps = list(my_bkps) + [len(X)]
        if my_bkps[-1] != len(X):
            my_bkps = list(my_bkps) + [len(X)]
        return score, my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        print(bkps,my_bkps)
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.outputs = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.params)
        self.scores = np.asarray([i[0] for i in self.outputs],dtype='object')
        self.reco_bkps = np.asarray([i[1] for i in self.outputs],dtype='object')
        self.n_bkps = np.asarray([len(i) for i in self.reco_bkps],dtype='object')
        
        
        # compute gain for the found bkps
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.asarray(self.gains,dtype='object')
        
        
        # select the best solution
        #print(self.reco_bkps)
        #print(self.gains)
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        #print(best_bkps)
        best_score = self.scores[self.gains == self.gains.max()][0]
        
        return best_score, best_bkps
    
    
    
from online_change_rulsif import ChangePointDetectionOnline_RuLSIF 

class OnlineRuLSIF(object):
    
    def __init__(self, lag_size=100, height=None, smooth=True):
        
        self.lag_size = lag_size
        self.height = height
        self.smooth = smooth
        
        self.params = list(itertools.product([1, 10], [1, 10], [1, 10], [0.01, 0.1], [1]))
    
    def run_predictions(self, X, params):
        algo = ChangePointDetectionOnline_RuLSIF(net='auto', scaler='auto', alpha=0.1, metric="None", lag_size=self.lag_size, 
                                         batch_size=params[0], step=params[1], n_epochs=params[2], lr=params[3], 
                                          periods=params[4] , lam=0.0, optimizer='Adam')
        # Detect change points
        score, my_bkps = algo.predict(X, height=self.height, smooth=self.smooth)
        if len(my_bkps) == 0:
            my_bkps = list(my_bkps) + [len(X)]
        if my_bkps[-1] != len(X):
            my_bkps = list(my_bkps) + [len(X)]
        return score, my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.outputs = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.params)
        self.scores = np.array([i[0] for i in self.outputs])
        self.reco_bkps = np.asarray([i[1] for i in self.outputs],dtype='object')
        self.n_bkps = np.asarray([len(i) for i in self.reco_bkps],dtype='object')
        
        
        # compute gain for the found bkps
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.asarray(self.gains,dtype='object')
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        best_score = self.scores[self.gains == self.gains.max()][0]
        
        return best_score, best_bkps
    
    
from algorithms import ChangePointDetectionRuLSIF

class OrigRuLSIF(object):
    
    def __init__(self, alpha=0.1, kernel_num=10, periods=1, window_size=100, 
                 step=10, n_runs=1, debug=0, height=None, smooth=True):
        
        self.cpd = ChangePointDetectionRuLSIF(alpha=alpha, kernel_num=kernel_num, periods=periods, 
                                              window_size=window_size, step=step, n_runs=n_runs, debug=debug)
        self.height = height
        self.smooth = smooth
        
    def predict(self, X, bkps=None):
        
        # Detect change points
        score, my_bkps = self.cpd.predict(X, height=self.height, smooth=self.smooth)
        if len(my_bkps) == 0:
            my_bkps = list(my_bkps) + [len(X)]
        if my_bkps[-1] != len(X):
            my_bkps = list(my_bkps) + [len(X)]
        return score, my_bkps

import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, n_in, n_out):
        
        super(NN, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_in, 2 * n_in)
        self.fc2 = nn.Linear(2 * n_in, 3 * n_in)
        self.fc3 = nn.Linear(3 * n_in, n_out)        
    
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        
        return x

class Contrastive_NN(object):

    def __init__(self, threshold, t_min=20, n_out_min=10, B=10, delta_max=50, n_epochs=200):
        #self.X = X,
        self.threshold = threshold
        self.t_min = 20
        self.n_out_min=10
        self.B =10
        self.delta_max = 50
        self.n_epochs = 20
        self.model = NN
        #print(self.n_out_min)

    def compute_test_stat_nn(self,X):
    
        #X = X.reshape(-1, 1)
        
        # Sample size
        n = X.shape[0]
        
        # Initialization
        T = np.zeros((n, n))
        
        stopping_time = -1
        #print(self.t_min,n)
        for t in range(self.t_min, n):
        
            #if (t % 100) == 0:
            #    print('Iteration', t)
                
            for tau in range(np.maximum(t - self.n_out_min - self.delta_max, self.n_out_min), t-self.n_out_min):
                
                # Initialize neural network
                f = self.model(n_in=X.shape[1], n_out=1)
                
                # Parameters of the optimizer
                opt = torch.optim.Adam(f.parameters(), lr=1e-1)
                
                X_t = torch.tensor(X[:t, :], dtype=torch.float32, requires_grad=True)
                
                # weights
                W = torch.cat((torch.ones(tau) * (t - tau), torch.ones(t - tau) * tau)).reshape(-1, 1)
                
                # Create "virtual" labels
                Y_t = torch.cat((torch.ones(tau), torch.zeros(t - tau))).reshape(-1, 1)
        
                # Loss function    
                loss_fn = nn.BCEWithLogitsLoss(weight=W)
                
                # Neural network training
                for epoch in range(self.n_epochs):
                    
                    loss = loss_fn(f(X_t), Y_t).mean()
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
                Z = f(X_t).detach().numpy().reshape(-1)
                
                # Use thresholding to avoid numerical issues
                Z = np.minimum(Z, self.B)
                Z = np.maximum(Z, -self.B)
                
                D = np.zeros(t)
                D[:tau] = 2 / (1 + np.exp(-Z[:tau]))
                D[tau:] = 2 / (1 + np.exp(Z[tau:]))
                D = np.log(D)
                
                # Compute statistics for each t
                # and each change point candidate tau
                T[tau, t] = tau * (t - tau) / t * (np.mean(D[:tau]) + np.mean(D[tau:]))
                
            if (np.max(T[:, t]) > self.threshold):
                
                stopping_time = t
                break
        
        # Array of test statistics
        S = np.max(T[:, :stopping_time + 1], axis=0)
        
        return S, stopping_time

    def predict(self,X,bkps=None):
        # Initialization
        st_nn = 0
        new_st_nn = 0

        # the threshold
        #z_nn = self.threshold

        # Initialization of the test statistic
        S_nn = np.empty(0)

        # Initialization of the list of detected change points
        change_points_nn = []


        # Initialization of the delays array and
        # the false alarms counter
        delays_nn = np.empty(0)
        current_change_point_ind = 0
        false_alarms_nn = 0

        data = X.copy()
        #print(data.shape)

        while new_st_nn >= 0:
            
            # Run the procedure until the moment
            # it reports a change point occurrence
            #print(data[st_nn + 1:])
            new_S_nn, new_st_nn = self.compute_test_stat_nn(data[st_nn + 1:])
            #print(new_S_nn,new_st_nn)
            
            S_nn = np.append(S_nn, new_S_nn)
            
            st_nn += new_st_nn
            change_points_nn += [int(st_nn)]

        return S_nn, change_points_nn


        