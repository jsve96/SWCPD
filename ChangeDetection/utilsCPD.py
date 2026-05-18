import numpy as np
import torch
from scipy.stats import norm,gamma
import pandas as pd

#shape rate notation for Gamma distribution
def mom_estimates(X):
    mean_X = np.mean(X)
    var_X = np.var(X, ddof=1)
    alpha_hat = mean_X**2 / var_X
    beta_hat = mean_X/var_X
    return alpha_hat, beta_hat



def estimate_parameter_CI(X,p):
    alpha_hat, beta_hat = mom_estimates(X)
    n= len(X)
    var_alpha_hat = (6* alpha_hat**2) / n  #+ beta_hat**4*var_X
    var_beta_hat = beta_hat**2/(n*alpha_hat) + beta_hat**6/alpha_hat**2*(2*alpha_hat**2)/beta_hat**4/n

    # Confidence level (e.g., 95%)
    confidence_level = p
    z_alpha_2 = norm.ppf(1 - (1 - confidence_level) / 2)  # Critical value for normal distribution

    # Confidence intervals
    alpha_CI = (alpha_hat - z_alpha_2 * np.sqrt(var_alpha_hat), alpha_hat + z_alpha_2 * np.sqrt(var_alpha_hat))
    beta_CI = (beta_hat - z_alpha_2 * np.sqrt(var_beta_hat), beta_hat + z_alpha_2 * np.sqrt(var_beta_hat))

    return alpha_CI,beta_CI


def bootstrap_mom_estimates(X, n_bootstrap=1000, random_seed=None):
    
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(X)
    alpha_bootstrap = np.empty(n_bootstrap)
    beta_bootstrap = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(X, size=n, replace=True)
        # Compute method of moments estimates for the bootstrap sample
        alpha_hat, beta_hat = mom_estimates(bootstrap_sample)
        alpha_bootstrap[i] = alpha_hat
        beta_bootstrap[i] = beta_hat
    
    alpha_ci = np.percentile(alpha_bootstrap, [2.5, 97.5])  # 95% CI for alpha
    beta_ci = np.percentile(beta_bootstrap, [2.5, 97.5])

    return alpha_ci[0], beta_ci[1]



class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, seq_len=10,split=0.5):
        self.X = X
        #self.y = y
        self.seq_len = seq_len
        self.split = split

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        window = index + int(self.seq_len*self.split)
        #print(index,window)
        #self.seq_len - int(self.seq_len//2)
        #print(window)
        return (self.X[index:window],self.X[window:window+ int(self.seq_len*(1-self.split))])
    


#def project_and_calc_dist_torch(X, Y, theta, p,device='cpu'):
   # X = X.to(device)
   # Y = Y.to(device)
   # theta = theta.to(device)
   # x_proj = torch.matmul(X, theta.T)
   # y_proj = torch.matmul(Y, theta.T)

   # qs = torch.linspace(0, 1, 100, device=device)

    # Compute quantiles for x_proj and y_proj
   # xp_quantiles = torch.quantile(x_proj, qs, dim=0, interpolation="lower")
   # yp_quantiles = torch.quantile(y_proj, qs, dim=0, interpolation="lower")

    # Calculate the p-distance between quantiles
   # dist_p = torch.abs(xp_quantiles - yp_quantiles)**p

   # return dist_p


def project_and_calc_dist_torch(X, Y, theta, p,device='cpu'):
    X ,Y, theta= X.to(device),Y.to(device), theta.T.to(device)

    x_proj = torch.matmul(X, theta)
    y_proj = torch.matmul(Y, theta)


    qs = torch.linspace(0, 1, 100, device=device)

    # Compute quantiles for x_proj and y_proj
    xp_quantiles = torch.quantile(x_proj, qs, dim=0, interpolation="linear")
    yp_quantiles = torch.quantile(y_proj, qs, dim=0, interpolation="linear")

    # Calculate the p-distance between quantiles
    dist_p = torch.pow(torch.abs(xp_quantiles - yp_quantiles),p)

    return dist_p


def sample_theta_torch(X, num_samples=10, device='cpu',seed=2025):
    """
    Samples random normalized vectors (theta) using PyTorch.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, d).
        num_samples (int): Number of random samples to generate.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Normalized random vectors of shape (num_samples, d).
    """
    _, d = X.shape
    # Generate random values
    torch.manual_seed(seed)
    theta = torch.randn(num_samples, d, device=device)
    # Compute the norm along the last dimension
    theta_norm = torch.norm(theta, dim=1, keepdim=True)
    # Normalize each vector
    theta_normed = theta / theta_norm
    return theta_normed


def gamma_conf_interval(step, a, b, confidence=0.95):
    alpha = step * a
    lower_bound = gamma.ppf((1 - confidence) / 2, alpha, scale=b)
    upper_bound = gamma.ppf(1 - (1 - confidence) / 2, alpha, scale=b)
    return lower_bound, upper_bound




def load_master_data(data_path="/datasets/has2023_master.csv.zip"):
    """
    Load the given CSV file containing the labelled challenge data.
    Returns a pandas DataFrame where each column is a sensor measurement
    or label and each row corresponds to a single time series.

    Parameters
    ----------
    data_path : str, default: "../datasets/has2023_master.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.

    Examples
    --------
    >>> data = load_master_data()
    >>> data.head()
    """
    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")


def load_data(data_path="../datasets/has2023.csv.zip"):
    """
    Load the given CSV file containing the sensor data for the challenge.
    Returns a pandas DataFrame where each column is a sensor measurement and
    each row corresponds to a single time series of sensor data.

    Parameters
    ----------
    data_path : str, default: "../datasets/has2023.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.

    Examples
    --------
    >>> data = load_data()
    >>> data.head()
    """
    np_cols = ["x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")


### from https://github.com/alan-turing-institute/TCPDBench/blob/master/analysis/scripts/metrics.py
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

from sklearn.metrics import auc

# def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
#     """Compute the F-measure based on human annotations.

#     annotations : dict from user_id to iterable of CP locations
#     predictions : iterable of predicted CP locations
#     alpha : value for the F-measure, alpha=0.5 gives the F1-measure
#     return_PR : whether to return precision and recall too

#     Remember that all CP locations are 0-based!

#     >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
#     1.0
#     >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
#     0.9090909090909091
#     >>> f_measure({1: [], 2: [10], 3: [50]}, [])
#     0.8
#     """
#     # ensure 0 is in all the sets
#     Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
#     for Tk in Tks.values():
#         Tk.add(0)

#     X = set(predictions)
#     X.add(0)

#     Tstar = set()
#     for Tk in Tks.values():
#         for tau in Tk:
#             Tstar.add(tau)

#     K = len(Tks)

#     P = len(true_positives(Tstar, X, margin=margin)) / len(X)

#     TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
#     R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

#     F = P * R / (alpha * R + (1 - alpha) * P)
#     if return_PR:
#         return F, P, R
#     return F, auc([0,R,1.0],[1.0,P,0])
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
    print(true_positives(Tstar, X, margin=margin))

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

from sklearn.cluster import KMeans

def get_feature_contribution(SL, theta,quantile_level,max=False,test=True):
    N = SL.shape[0]
    q_ind = int(np.floor(N*quantile_level))
    if max:
        return np.abs(theta[np.argsort(SL)[-1],:])
    thetas = theta[np.argsort(SL)[q_ind:],:]
    score = np.abs(thetas).mean(axis=0)
    #test= True
    if test:
        kmeans= KMeans(n_clusters=2,random_state=0)
        labels = kmeans.fit_predict(theta[np.argsort(SL)[q_ind:],:])
        centers = kmeans.cluster_centers_
        #print(centers)
        return centers[0]
    #print(score)
    return score


def remove_important_features_syn(X, Y, num_features_to_remove,N_Theta,max_parameter=False,q=0.95,p=2,device='cpu'):
    
    """
    Iteratively remove the most important features from X and Y
    and calculate new contributions at each step.
    
    X, Y: Data matrices with d columns (features).
    num_features_to_remove: Number of features to remove step by step.
    
    Returns:
    - A list of dictionaries containing removed feature index and new contributions.
    """
    betas = []
    removed_features = []  # To store removed feature indices and their contributions
    SWDs = []
    step = 1
    Y_imp = Y.clone().to(device)
    Contributions_out = []
    #THETA = sample_theta(X,N_Theta)
    THETA = sample_theta_torch(X,N_Theta,device=device)
    for _ in range(num_features_to_remove):
        #THETA = sample_theta_torch(X,N_Theta,device=device)
        dist = project_and_calc_dist_torch(X,Y_imp,THETA,p=p,device=device).mean(axis=0).detach().cpu().numpy()
        SWDs.append(dist.mean(axis=0))
        #print(SWDs)
        #print('removal: {}'.format(step))
        #print('SWD: {}'.format(dist.mean(axis=0)))
        #print('beta estimate: {}'.format(dist.mean(axis=0)/dist.var(axis=0,ddof=1)))
        betas.append(dist.mean(axis=0)/dist.var(axis=0,ddof=1))
        #print('alpha estimate: {}'.format(dist.mean(axis=0)**2/dist.var(axis=0,ddof=1)))
        # 1. Calculate feature contributions for the remaining features
        contributions = get_feature_contribution(dist,THETA.detach().cpu().numpy(),q,max=max_parameter,test=False)
        #print(contributions)
        # 2. Find the feature with the highest contribution
        max_contrib_index = np.argmax(contributions)
        #print(max_contrib_index)
        max_contrib_value = contributions[max_contrib_index]
        #print(contributions)
        # 3. Store the feature removal information
        removed_features.append({
            'removed_feature': max_contrib_index,
            'contribution_value': max_contrib_value,
            'beta': betas[-1],
            'alpha': dist.mean(axis=0)**2/dist.var(axis=0,ddof=1) 
        })
        # 4. Replace the feature from Y with mean value of X

        Y_imp[:,max_contrib_index] = X[:,max_contrib_index]#torch.ones(Y.shape[0]).to(device)*X[:,max_contrib_index].mean()
        Contributions_out.append(contributions)
        step+=1
    return removed_features,Contributions_out




def update_index_map(ind_map, removed_ind):
    if removed_ind == list(ind_map.keys())[-1]:
        del ind_map[removed_ind]
    else:
        for k, v in list(ind_map.items())[:-1]:
            if k>= removed_ind:
                ind_map[k] = ind_map[k+1]
        del ind_map[k+1]
    return ind_map

def remove_important_features_syn_ablation(X, Y, num_features_to_remove,N_Theta,q=0.95,p=2,device='cpu'):
    
    """
    Iteratively remove the most important features from X and Y
    and calculate new contributions at each step.
    
    X, Y: Data matrices with d columns (features).
    num_features_to_remove: Number of features to remove step by step.
    
    Returns:
    - A list of dictionaries containing removed feature index and new contributions.
    """
    betas = []
    removed_features = []  # To store removed feature indices and their contributions
    SWDs = []
    step = 1
    Y_imp = Y.clone().to(device)
    Contributions_out = []
    ind_map = {i:i for i in range(X.shape[1])}

    #THETA = sample_theta(X,N_Theta)
    #THETA = sample_theta_torch(X,N_Theta,device=device)
    for _ in range(num_features_to_remove):
        #print(X.shape[1])
        ind_left = list(np.arange(X.shape[1]))
        #print(ind_left)
        THETA = sample_theta_torch(X,N_Theta,device=device)
        dist = project_and_calc_dist_torch(X,Y_imp,THETA,p=p,device=device).mean(axis=0).detach().cpu().numpy()
        SWDs.append(dist.mean(axis=0))
        #print(SWDs)
        
        betas.append(dist.mean(axis=0)/dist.var(axis=0,ddof=1))
        
        # 1. Calculate feature contributions for the remaining features
        contributions = get_feature_contribution(dist,THETA.detach().cpu().numpy(),q,max=True,test=False)
        # 2. Find the feature with the highest contribution
        max_contrib_index = np.argmax(contributions)
        print(f"argmax {max_contrib_index}")
        max_contrib_value = contributions[max_contrib_index]
        print(ind_map[max_contrib_index])
        #print(contributions)
        # 3. Store the feature removal information
        removed_features.append({
            'removed_feature': ind_map[max_contrib_index],
            'contribution_value': max_contrib_value,
            'beta': betas[-1],
            'alpha': dist.mean(axis=0)**2/dist.var(axis=0,ddof=1) 
        })
        ind_map = update_index_map(ind_map,max_contrib_index)
        # 4. Replace the feature from Y with mean value of X
        ind_left.remove(max_contrib_index)

        Y_imp = Y_imp[:,ind_left]
        X = X[:,ind_left]#torch.ones(Y.shape[0]).to(device)*X[:,max_contrib_index].mean()
        Contributions_out.append(contributions)
        step+=1
    return removed_features,Contributions_out

