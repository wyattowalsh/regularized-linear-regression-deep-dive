from src import utilities as utils
from src import linear_regression as regress

import numpy as np
import pandas as pd

def cross_validation_indices(df, k=5):
    """This function uses K-Fold C.V. to choose tuning parameters for 
    the different regression techniques. It not only returns the 
    best parameters, but also the parameters tested, the coeefficients 
    associated with them, the minimum error, and the error for each 
    parameter for the last fold."""
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    sets = np.array_split(indices, k)
    return sets, indices

def cross_validation_parameter_grid(train, regression_fn, n = 100):


def cross_validation_error(df, regression_fn, n, k=5):
    if regression_fn == ols:
        continue
    elif regression_fn == ridge:
        
    elif regression_fn == lasso: 

    sets, indices = cross_validation_indices(df, k)
    error = np.zeros(k)
    B = np.zeros((n, k, len(df.columns)))
    for i in np.arange(k):
        for j in np.arange(n):
            train_indices = [p for p in indices if p not in sets[i]]
            train_data = df.values[train_indices,:]
            B[j, i, :] = regression_fn(train_data)
            test_data = df.values[sets[i],:]
            y_hat = np.dot(test_data[:,0:-1],B[j,i, 1:])+B[j,i,0]
            error[i] = np.linalg.norm(test_data[:,-1]-y_hat)**2
    error_mean = sum(error)/k
    return error_mean, error, B


def cross_validation(df, n, k=5,type = 'ridge'):
    """This function uses K-Fold C.V. to choose tuning parameters for 
    the different regression techniques. It not only returns the 
    best parameters, but also the parameters tested, the coeefficients 
    associated with them, the minimum error, and the error for each 
    parameter for the last fold."""
    np.random.seed(18)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    sets = np.array_split(indices, k)
    
    if type == 'ols':
        error = 0
        for i in np.arange(k):
            train_indices = [p for p in indices if p not in sets[i]]
            train_data = df.iloc[train_indices,:]
            test_data = df.iloc[sets[i],:]
            B = ols(train_data)
            y_hat = np.dot(test_data.iloc[:,0:-1],B[1:])+B[0]
            error += np.linalg.norm(test_data.iloc[:,-1]-y_hat)**2
        error = (1/k)*error
        return error
    
    elif type == 'ridge' :
        u = np.geomspace(1e-2,1e5,n)
        error = np.zeros(n)
        coeff = np.zeros((n,11))
        for i in np.arange(k):
            for j in np.arange(len(u)):
                train_indices = [p for p in indices if p not in sets[i]]
                train_data = df.iloc[train_indices,:]
                test_data = df.iloc[sets[i],:]
                B = ridge_regression(train_data, u[j])
                coeff[j,:] = B[1:]
                y_hat = np.dot(test_data.iloc[:,0:-1],B[1:])+B[0]
                error[j] = error[j] + np.linalg.norm(test_data.iloc[:,-1]-y_hat)**2
        error = (1/k)*error
        z = np.argmin(error)
        u_star = u[z]
        return u_star,min(error),error,u, coeff
            
    elif type == 'lasso':
        error = np.zeros(n)
        for i in np.arange(k):
            train_indices = [p for p in indices if p not in sets[i]]
            train_data = df.iloc[train_indices,:]
            test_data = df.iloc[sets[i],:]
            train_data_std = standardize(train_data)
            X = train_data_std.iloc[:,0:-1].values
            Y = train_data_std.iloc[:,-1].values
            u_max = max(list(abs(np.dot(np.transpose(X),Y)))) / len(train_data_std)
            u_ = np.geomspace(u_max,u_max*0.0001,n)
            B = lasso_regression_covar(train_data,1,u_ )[1]
            for j in np.arange(len(u_)):
                B_ = B[j]
                y_hat =  B_[0]+np.dot(standardize(test_data.iloc[:,0:-1]),
                                      B_[1:])
                error[j] = error[j] + np.linalg.norm(test_data.iloc[:,-1]-
                                                 y_hat)**2
        error = (1/k)*error
        z = np.argmin(error)
        u_star = u_[z]
        return u_star,min(error),error,u_,B
            
    elif type == 'elastic net' or 'elasticnet' or 'elastic_net' or 'elastic-net':
        error = np.zeros((n,n))
        alpha =  np.geomspace(1,1e-6,n)
        tot_coeff = np.zeros((n,n,12))
        tot_u = np.zeros((n,n))
        for i in np.arange(k):
            train_indices = [p for p in indices if p not in sets[i]]
            train_data = df.iloc[train_indices,:]
            test_data = df.iloc[sets[i],:]
            train_data_std = standardize(train_data)
            X = train_data_std.iloc[:,0:-1].values
            Y = train_data_std.iloc[:,-1].values
            for j in np.arange(n):
                u_max = max(list(abs(np.dot(np.transpose(X),Y)))) / len(train_data_std)/(alpha[j])
                u = np.geomspace(u_max,u_max*0.001,n)
                tot_u[j,:] = u 
                B = elastic_net(train_data,1,alpha[j],u)[1] 
                tot_coeff[j,:,:] = B
                for m in np.arange(n):
                    B_ = B[m]
                    y_hat = np.dot(standardize(test_data.iloc[:,0:-1]),
                                   B_[1:])+B_[0]
                    error[j,m] = error[j,m] + np.linalg.norm(test_data.iloc[:,-1])
        error = (1/k)*error
        indice = np.unravel_index(error.argmin(),error.shape)
        alpha_star = alpha[indice[0]]
        u_star = u[indice[1]]
        return u_star, alpha_star, min(error.flatten()), error, tot_u,alpha,tot_coeff
    else:
        print("wrong command; must be: ridge, lasso, or elastic net")