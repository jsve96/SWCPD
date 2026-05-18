import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time


def mean_std_report(mean, std, cols, n_digits=3):
    report = pd.DataFrame()
    for acol in cols:
        new_col_val = []
        mean_col = mean[acol].values
        std_col = std[acol].values
        for i in range(len(mean_col)):
            i_mean = np.round(mean_col[i], n_digits)
            i_std = np.round(std_col[i], n_digits)
            mean_std = str(i_mean) + ' pm ' + str(i_std)
            new_col_val.append(mean_std)
        report[acol] = new_col_val
    return report


from sklearn.preprocessing import StandardScaler
import ruptures.metrics
from copy import copy, deepcopy
import inspect


def get_label(L):
    L_new = [0]
    for i in range(1, len(L)):
        if L[i] == L[i-1]:
            L_new.append(0)
        else:
            L_new.append(1)
    return np.array(L_new)


def get_files_list(path):
    files = []
    for aname in os.listdir(path):
        if (aname[-3:] == "csv") or (aname[-3:] == "txt"): 
            files.append(os.path.join(path, aname))
    return files


def make_downsample(X, L, jump):
    states = np.cumsum(L)
    states_down = states[::jump]
    X_down      = X[::jump]
    L_down      = get_label(states_down)
    return X_down, L_down


from sklearn.preprocessing import StandardScaler
import ruptures.metrics
from test_utils import *
from sklearn.metrics import auc



def detection_delay(gt, pred):
    """Compute detection delay for each predicted CP.
    
    Args:
        T (set): Ground truth CPs (annotations).
        X (set): Predicted CPs.

    Returns:
        delays (dict): {prediction: detection delay}
        avg_delay (float): Mean detection delay
    """
    T = set(gt)
    X = set(pred)
    delays = {}
    for x in X:
        if T:  # Ensure T is not empty
            closest_t = min(T, key=lambda t: abs(x - t))  # Find closest annotation
            delays[x] = abs(x - closest_t)  # Compute delay
    
    avg_delay = np.mean(list(delays.values())) if delays else 0  # Mean delay
    return delays, avg_delay


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too

    Remember that all CP locations are 0-based!

    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    TP = false_positives(Tstar,X,margin=margin)
    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F, auc([0,R,1.0],[1.0,P,0]),len(false_positives(Tstar,X,margin=margin))



def true_positives(T, X, margin=5):
    """Compute true positives without double counting

    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP

# def false_positives(T, X, margin=5):
#     """Returns a set of false positives (incorrectly predicted CPs)."""
#     TP = true_positives(T, X, margin)
#     return X - TP  # FP = All Predictions - True Positives
def false_positives(T, X, margin=5):
    """Compute false positives (predictions that don't match any true CPs within margin)."""
    FP = set()
    for x in X:
        if not any(abs(x - t) <= margin for t in T):  # No match within margin
            FP.add(x)
    return FP

def overlap(A, B):
    """Return the overlap (i.e. Jaccard index) of two sets

    >>> overlap({1, 2, 3}, set())
    0.0
    >>> overlap({1, 2, 3}, {2, 5})
    0.25
    >>> overlap(set(), {1, 2, 3})
    0.0
    >>> overlap({1, 2, 3}, {1, 2, 3})
    1.0
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.

    >>> partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> partition_from_cps([1,2,7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    """Compute the covering of a segmentation S by a segmentation Sprime.

    This follows equation (8) in Arbaleaz, 2010.

    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2, 3}, {4, 5}, {6}])
    0.8333333333333334
    >>> cover_single([{1, 2, 3, 4, 5, 6}], [{1, 2, 3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2}, {3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1}, {2}, {3}, {4, 5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.3333333333333333
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """Compute the average segmentation covering against the human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series

    >>> covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45)
    0.7962962962962963
    >>> covering({1: [], 2: [10], 3: [40]}, [10], 45)
    0.7954144620811286
    >>> covering({1: [], 2: [10], 3: [40]}, [], 45)
    0.8189300411522634

    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def detection_delay(gt, pred):
    """Compute detection delay for each predicted CP.
    
    Args:
        T (set): Ground truth CPs (annotations).
        X (set): Predicted CPs.

    Returns:
        delays (dict): {prediction: detection delay}
        avg_delay (float): Mean detection delay
    """
    T = set(gt)
    X = set(pred)
    delays = {}
    for x in X:
        if T:  # Ensure T is not empty
            closest_t = min(T, key=lambda t: abs(x - t))  # Find closest annotation
            delays[x] = abs(x - closest_t)  # Compute delay
    
    avg_delay = np.mean(list(delays.values())) if delays else 0  # Mean delay
    return delays, avg_delay

class Tester(object):
    
    def __init__(self):
        pass

    def run(self, afile, model, downsample=1, margin=50, verbose=0, sigma=0,labels=True,seed=0):

        if verbose > 0: print("File: ", afile)

        # read data
        data = pd.read_csv(afile, index_col=False)

        # decompose data
        if isinstance(labels,bool):
            label = data["Label"].values
            #label = np.append(label, X.shape[0])
            X = data.drop(columns=["Time", "Label"]).values
        else:
            try:
                X = data.drop(columns=["Time"]).values
            except:
                X = data.values
        
            
        X = np.nan_to_num(X)
        if downsample > 1:
            X, label = make_downsample(X, label, downsample)
        T = np.arange(len(X))

        # apply standard scaler
        X = StandardScaler().fit_transform(X)
        if verbose > 0: print("X shape: ", X.shape)

        # get true CP ids
        if isinstance(labels,bool):
            y_true = list(T[label == 1]) + [len(X)]
            #y_true = y_true
        else: 
            y_true = list(labels)
            y_true.append(len(X))
            print(y_true)
        if verbose > 0: print("y_true: ", y_true)
        
        # add noise
        if sigma > 0:
            np.random.seed(seed)
            X += np.random.normal(0, sigma, X.shape) 
        X = StandardScaler().fit_transform(X)

        # run CPD model
        start =time.time()
        score, y_pred = model.predict(X, y_true)
        end = time.time()


        if verbose > 0: print("y_pred: ", y_pred)

        # quality metrics
        #ri = ruptures.metrics.randindex(y_true, y_pred)
        #precision, recall = ruptures.metrics.precision_recall(y_true, y_pred, margin=margin)
        #f1 = 2 * precision * recall / (precision + recall + 10**-6)
        F1, AUCs, FPs = [],[],[]
        for m in [margin]:#[10,20,30,40,50]: #[50,100,150,200,250]
            f1,AUC,FP = f_measure({'0':list(y_true[:-1])},list(y_pred[:-1]),margin=m)
            F1.append(f1)
            AUCs.append(AUC)
            FPs.append(FP)
        self.T = T
        self.X = X
        try:
            self.label = label
        except:
            self.label = y_true
        self.bkps = y_true
        
        self.score = score
        self.my_bkps = y_pred
        print(y_pred)
        #self.RI = ri
        self.F1 = F1
        #self.precision = precision
        #self.recall = recall
        self.model = model
        self.AUC = AUCs
        self.FP = FPs
        self.Thresholds = [margin]#[10,20,30,40,50]
        self.dd = detection_delay(list(y_true),y_pred)[1]
        self.covering = covering({0: y_true}, y_pred, X.shape[0])
        self.runtime = end-start
        return


def run_test_on_dataset(dir_path, model, downsample=1, margin=50, verbose=0, sigma=0,seed=0):
    
    if verbose > 0: print("Dataset: ", dir_path)
    
    # get list of files in dataset
    files = get_files_list(dir_path)
    files.sort()
    
    # fataframe for quality metrics

    ##margin 10 ,20, 30 ,40, 50

    reports = [pd.DataFrame(columns=["F1", "AUC", "FP","DD","Threshold","Covering","Runtime"]) for i in range(1)]
    #print(reports)

    
    # run test on each file
    for afile in files:
        print(afile)
        target = afile.split("/")[-1].split(".")[0]+".npy"
        try:
            label = np.load(dir_path+"/"+target)
            #print(label)
        except:
            label = True
        t = Tester()
        print(label)
        t.run(afile, copy(model), downsample, margin, verbose, sigma,labels=label,seed=seed)
        #print(t.AUC)
        for i in range(len(t.Thresholds)):
        
            reports[i].loc[len(reports[i])] = [t.F1[i], t.AUC[i], t.FP[i],t.dd, t.Thresholds[i],t.covering,t.runtime]
    
    return [pd.concat([report.mean(), report.std(), report.min(),report.max()], axis=1) for report in reports]