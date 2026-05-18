import numpy as np
import pandas as pd

def sanity_check(matrix):
    if not np.allclose(np.linalg.norm(matrix,axis=1),1):
        raise ValueError("At least one non unit vector")
    return True

def project_and_calc_dist(X,Y,theta,p):
    x_proj = np.dot(X, theta.T)
    y_proj = np.dot(Y, theta.T)
    qs = np.linspace(0,1,100)
    xp_quantiles = np.quantile(x_proj, qs, axis=0, method="inverted_cdf")
    yp_quantiles = np.quantile(y_proj, qs, axis=0, method="inverted_cdf")
    dist_p = np.abs(xp_quantiles - yp_quantiles)**p
    return dist_p


def activation(vector):
    return np.exp(vector)/np.exp(vector).sum()


def sample_theta(X,num_smaples=10):
    _ , d = X.shape
    theta = np.random.randn(num_smaples,d)
    theta_norm = np.linalg.norm(theta, axis=1)
    theta_normed = theta / theta_norm[:, np.newaxis]
    return theta_normed



def get_mu_var(X):
    ### input array NXL
    N,d = X.shape
    mu_norm = np.linalg.norm(X.mean(axis=0),axis=0)**2/d
    cov = pd.DataFrame(X).cov().values
    trace = np.trace(cov)/d
    return mu_norm, trace

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
        print(centers)
        return centers[0]
    return score


def remove_important_features(X, Y, num_features_to_remove,N_Theta):
    
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
    Y_imp = Y.copy()
    #THETA = sample_theta(X,N_Theta)
    for _ in range(num_features_to_remove):
        #THETA = sample_theta(X,N_Theta)
        #print(THETA.shape)
        THETA = np.eye(X.shape[1])
        dist = project_and_calc_dist(X,Y_imp,THETA,p=2).mean(axis=0)
        SWDs.append(dist.mean(axis=0))
        #print('removal: {}'.format(step))
        #print('SWD: {}'.format(dist.mean(axis=0)))
        #print('beta estimate: {}'.format(dist.mean(axis=0)/dist.var(axis=0,ddof=1)))
        betas.append(dist.mean(axis=0)/dist.var(axis=0,ddof=1))
        #print('alpha estimate: {}'.format(dist.mean(axis=0)**2/dist.var(axis=0,ddof=1)))
        # 1. Calculate feature contributions for the remaining features
        contributions = get_feature_contribution(dist,THETA,0.99,test=False)
        
        # 2. Find the feature with the highest contribution
        max_contrib_index = np.argmax(contributions)
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
        Y_imp[:,max_contrib_index] = np.ones(Y.shape[0])*X[:,max_contrib_index]
        step+=1
    #plt.plot(range(len(betas)),betas,marker='.')
    return removed_features,betas,SWDs


def remove_important_features_syn(X, Y, num_features_to_remove,N_Theta,max_parameter=False,q=0.95):
    
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
    Y_imp = Y.copy()
    Contributions_out = []
    #THETA = sample_theta(X,N_Theta)
    for _ in range(num_features_to_remove):
        THETA = sample_theta(X,N_Theta)
        dist = project_and_calc_dist(X,Y_imp,THETA,p=2).mean(axis=0)
        SWDs.append(dist.mean(axis=0))
        #print('removal: {}'.format(step))
        print('SWD: {}'.format(dist.mean(axis=0)))
        #print('beta estimate: {}'.format(dist.mean(axis=0)/dist.var(axis=0,ddof=1)))
        betas.append(dist.mean(axis=0)/dist.var(axis=0,ddof=1))
        #print('alpha estimate: {}'.format(dist.mean(axis=0)**2/dist.var(axis=0,ddof=1)))
        # 1. Calculate feature contributions for the remaining features
        contributions = get_feature_contribution(dist,THETA,q,max=max_parameter)
        
        # 2. Find the feature with the highest contribution
        max_contrib_index = np.argmax(contributions)
        max_contrib_value = contributions[max_contrib_index]
        print(contributions)
        # 3. Store the feature removal information
        removed_features.append({
            'removed_feature': max_contrib_index,
            'contribution_value': max_contrib_value,
            'beta': betas[-1],
            'alpha': dist.mean(axis=0)**2/dist.var(axis=0,ddof=1) 
        })
        # 4. Replace the feature from Y with mean value of X
        Y_imp[:,max_contrib_index] = np.ones(Y.shape[0])*X[:,max_contrib_index].mean()
        Contributions_out.append(contributions)
        step+=1
    #plt.plot(range(len(betas)),betas,marker='.')
    return removed_features,betas,SWDs,Contributions_out


def cosin_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))