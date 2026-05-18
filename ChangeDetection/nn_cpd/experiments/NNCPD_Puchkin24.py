import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cvx
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import block_diag


def compute_multivariate_design(X):
    
    Psi = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    
    return Psi


def compute_theta(Psi, tau):
    
    # Sample size
    t = Psi.shape[0]
    
    # Create "virtual" labels
    Y = np.append(np.ones(tau), -np.ones(t - tau))
    
    lr = LogisticRegression(penalty='none', fit_intercept=False, tol=5e-2,\
                            solver='newton-cg', class_weight='balanced', n_jobs=-1)
    lr.fit(Psi, Y)
    theta = (lr.coef_).reshape(-1)
    
    return theta


def compute_test_stat_linear(X, threshold, t_min=20, n_out_min=10, B=10, delta_max=50):
    
    # Sample size
    n = X.shape[0]
    
    # Compute design matrix
    Psi = compute_multivariate_design(X)
    
    # Initialization
    T = np.zeros((n, n))
    
    stopping_time = -1
    
    for t in range(t_min, n):
        
        #if (t % 100) == 0:
        #    print('Iteration', t)
            
        D = np.zeros(t)
        
        for tau in range(np.maximum(t - n_out_min - delta_max, n_out_min), t - n_out_min):
            
            # Compute the best fitting parameter theta
            theta = compute_theta(Psi[:t, :], tau)
            Z = Psi[:t, :] @ theta
            
            # Use thresholding to avoid numerical issues
            Z = np.minimum(Z, B)
            Z = np.maximum(Z, -B)
            
            D[:tau] = 2 / (1 + np.exp(-Z[:tau]))
            D[tau:] = 2 / (1 + np.exp(Z[tau:]))
            D = np.log(D)
            
            # Compute statistics for each t
            # and each change point candidate tau
            T[tau, t] = tau * (t - tau) / t * (np.mean(D[:tau]) + np.mean(D[tau:]))
        
        if np.max(T[:, t]) > threshold:
            
            stopping_time = t
            break
            
    # Array of test statistics
    S = np.max(T[:, :stopping_time + 1], axis=0)
    
    return S, stopping_time


class NN(nn.Module):
    def __init__(self, n_in, n_out):
        
        super(NN, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_in, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, n_out)        
    
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        
        return x
  


def compute_test_stat_nn(X, threshold, t_min=20, n_out_min=10, B=10, delta_max=50, n_epochs=200, model=NN):
    
    #X = X.reshape(-1, 1)
    
    # Sample size
    n = X.shape[0]
    
    # Initialization
    T = np.zeros((n, n))
    
    stopping_time = -1
    
    for t in range(t_min, n):
    
        #if (t % 100) == 0:
        #    print('Iteration', t)
            
        for tau in range(np.maximum(t - n_out_min - delta_max, n_out_min), t-n_out_min):
            
            # Initialize neural network
            f = model(n_in=X.shape[1], n_out=1)
            
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
            for epoch in range(n_epochs):
                
                loss = loss_fn(f(X_t), Y_t).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                
            Z = f(X_t).detach().numpy().reshape(-1)
            
            # Use thresholding to avoid numerical issues
            Z = np.minimum(Z, B)
            Z = np.maximum(Z, -B)
            
            D = np.zeros(t)
            D[:tau] = 2 / (1 + np.exp(-Z[:tau]))
            D[tau:] = 2 / (1 + np.exp(Z[tau:]))
            D = np.log(D)
            
            # Compute statistics for each t
            # and each change point candidate tau
            T[tau, t] = tau * (t - tau) / t * (np.mean(D[:tau]) + np.mean(D[tau:]))
            
        if (np.max(T[:, t]) > threshold):
            
            stopping_time = t
            break
       
    # Array of test statistics
    S = np.max(T[:, :stopping_time + 1], axis=0)
    
    return S, stopping_time