import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split
from copy import deepcopy
from densratio import densratio

from algorithms import KL, KL_sym, JSD, PE, PE_sym, Wasserstein, roc_auc_score
from algorithms import autoregression_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sklearn.preprocessing import StandardScaler
from helper import SMA
from scaler import MockScaler, SmaScalerCache


from scipy import interpolate
from scipy.signal import find_peaks

def unified_score(T, T_score, score):
    uni_score = np.zeros(len(T))
    inter = interpolate.interp1d(T_score, score, kind='previous', fill_value=(0, 0), bounds_error=False)
    uni_score = inter(T)
    return uni_score


class BaseNN(nn.Module):
    def __init__(self, n_inputs=1):
        super(BaseNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 20)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class ChangePointDetectionOnline_RuLSIF(object):
    
    def __init__(self, net='auto', scaler='auto', alpha=0.1, metric="None", batch_size=1, periods=10, lag_size=100, step=1, 
                 n_epochs=1, lr=0.1, lam=0., optimizer="RMSprop", debug=0):
        
        if net == 'auto':
            self.base_net = BaseNN
        else:
            self.base_net = net
            
        if scaler is None:
            self.scaler = MockScaler(0)
        if scaler == 'auto':
            self.scaler = SmaScalerCache(lag_size + 2*batch_size)
        else:
            self.scaler = scaler
        
        self.net1 = None
        self.net2 = None
        self.alpha = alpha
        self.metric = metric
        self.batch_size = batch_size
        self.periods = periods
        self.lag_size = lag_size
        self.step = step
        self.n_epochs = n_epochs
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug

            
    def predict(self, X, distance=5, height=None, smooth=False):
        
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = self.reference_test(X_auto)
        
        self.net1 = self.base_net(X_auto.shape[1])
        self.net2 = self.base_net(X_auto.shape[1])
        
        self.criterion = nn.BCELoss()
        if self.optimizer == "Adam":
            self.opt1 = torch.optim.Adam(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.Adam(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            self.opt1 = torch.optim.SGD(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.SGD(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "RMSprop":
            self.opt1 = torch.optim.RMSprop(self.net1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.RMSprop(self.net2.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "ASGD":
            self.opt1 = optim.ASGD(self.net1.parameters(), lr=self.lr, lambd=0.0, alpha=0.75, t0=0.0, weight_decay=self.lam)
            self.opt2 = optim.ASGD(self.net2.parameters(), lr=self.lr, lambd=0.0, alpha=0.75, t0=0.0, weight_decay=self.lam)
        else:
            self.opt1 = torch.optim.Adam(self.model1.parameters(), lr=self.lr, weight_decay=self.lam)
            self.opt2 = torch.optim.Adam(self.model2.parameters(), lr=self.lr, weight_decay=self.lam)
        
        
        scores = [self.reference_test_predict(reference[i], test[i]) for i in range(len(reference))]
        T_scores = np.array([T[i] for i in range(len(reference))])
        
        T = np.arange(len(X))
        scores = unified_score(T, T_scores-self.step, scores)
        
        scores = SMA(scores, self.lag_size+self.batch_size)
        
        shift = self.lag_size + self.batch_size
        scores = unified_score(T, T-shift, scores)
        
        if smooth:
            from scipy.signal import savgol_filter
            width = int((np.round(0.25 * self.lag_size) // 2) * 2 + 1)
            scores = savgol_filter(scores, width, 1)
        
        width = 0.25 * (self.lag_size+self.batch_size)
        peaks, _ = find_peaks(scores, distance=distance, width=width, height=height)
        
        return np.array(scores), peaks
    
    
    def reference_test_predict(self, X_ref, X_test):
        
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        
        X = self.scaler.fit_transform(X)
        
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        
        n_last = min(self.batch_size, self.step)
        self.net1.train(False)
        test_preds = self.net1(X[y == 1][-n_last:]).detach().numpy()
        
        self.net2.train(False)
        ref_preds  = self.net2(X[y == 0][-n_last:]).detach().numpy()
        
        self.net1.train(True)
        self.net2.train(True)
        for epoch in range(self.n_epochs):  # loop over the dataset multiple times
            
            # forward + backward + optimize
            y_pred_batch = self.net1(X).squeeze()
            y_pred_batch_ref  = y_pred_batch[y == 0]
            y_pred_batch_test = y_pred_batch[y == 1]
            loss = 0.5 * (1 - self.alpha) * (y_pred_batch_ref**2).mean() + \
                   0.5 *      self.alpha  * (y_pred_batch_test**2).mean() - (y_pred_batch_test).mean()
            
            # set gradients to zero
            self.opt1.zero_grad()
            loss.backward()
            self.opt1.step()
            
            
            # forward + backward + optimize
            y_pred_batch = self.net2(X).squeeze()
            y_pred_batch_ref  = y_pred_batch[y == 1]
            y_pred_batch_test = y_pred_batch[y == 0]
            loss = 0.5 * (1 - self.alpha) * (y_pred_batch_ref**2).mean() + \
                   0.5 *      self.alpha  * (y_pred_batch_test**2).mean() - (y_pred_batch_test).mean()
            
            # set gradients to zero
            self.opt2.zero_grad()
            loss.backward()
            self.opt2.step()
            
            

        score = (0.5 * np.mean(test_preds) - 0.5) + (0.5 * np.mean(ref_preds) - 0.5)
            
        return score
    
    
    def reference_test(self, X):
        N = self.lag_size
        ws = self.batch_size
        T = []
        reference = []
        test = []
        for i in range(2*ws+N-1, len(X), self.step):
            T.append(i)
            reference.append(X[i-2*ws-N+1:i-ws-N+1])
            test.append(X[i-ws+1:i+1])
        return np.array(T), np.array(reference), np.array(test)
