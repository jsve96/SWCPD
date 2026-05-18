import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split
from copy import deepcopy
#from densratio import densratio

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
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class ChangePointDetectionOnline(object):
    
    def __init__(self, net='auto', scaler='auto', metric="KL", batch_size=1, periods=10, lag_size=100, step=1, 
                 n_epochs=1, lr=0.1, lam=0., optimizer="RMSprop", debug=0):
        torch.manual_seed(0)
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
        
        self.net = None
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
        
        
    
    
    def reference_test_predict(self, X_ref, X_test):
        
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        
        X = self.scaler.fit_transform(X)
        
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        
        self.net.train(False)
        n_last = min(self.batch_size, self.step)
        ref_preds  = self.net(X[y == 0][-n_last:]).detach().numpy()
        test_preds = self.net(X[y == 1][-n_last:]).detach().numpy()
        
        self.net.train(True)
        for epoch in range(self.n_epochs):  # loop over the dataset multiple times
            
            # forward + backward + optimize
            outputs = self.net(X)
            loss = self.criterion(outputs.squeeze(), y)
            
            # set gradients to zero
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        if self.metric == "KL_sym":
            score = KL_sym(ref_preds, test_preds)
        elif self.metric == "KL":
            score = KL(ref_preds, test_preds)
        elif self.metric == "JSD":
            score = JSD(ref_preds, test_preds)
        elif self.metric == "PE":
            score = PE(ref_preds, test_preds)
        elif self.metric == "PE_sym":
            score = PE_sym(ref_preds, test_preds)
        elif self.metric == "W":
            score = Wasserstein(ref_preds, test_preds)
        else:
            score = 0
            
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
    
    
    def predict(self, X, distance=5, height=None, smooth=False):
        
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = self.reference_test(X_auto)
        
        self.net = self.base_net(X_auto.shape[1])
        
        self.criterion = nn.BCELoss()
        if self.optimizer == "Adam":
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "RMSprop":
            self.opt = torch.optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "ASGD":
            self.opt = optim.ASGD(self.net.parameters(), lr=self.lr, lambd=0.0, alpha=0.75, t0=0.0, weight_decay=self.lam)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        
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
