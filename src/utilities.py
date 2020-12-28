# This is a script of utility functions utilized throughout the project

# Necessary imports
from math import *
import numpy as np
from src import linear_regression as lr
import time

import pandas as pd

from sklearn.linear_model import LinearRegression as skols
from sklearn.linear_model import Ridge as skridge
from sklearn.linear_model import Lasso as sklasso
from sklearn.linear_model import ElasticNet as skelasticnet
### Useful functions ###

def test_train_split(df, proportion_train):
	"""This function splits a given dataset into training and testing sets. 
	This is accomplished through the use of random sampling and indexing"""

	shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
	split = floor(len(shuffled) * proportion_train)
	train = shuffled.iloc[:split,]
	test = shuffled.iloc[split:,].reset_index(drop=True)
	return (train, test)

def standardize(train, test):
    """Uses the equation for standardization 
    {(value - mean(value))/(sd of value)} on the predictors.
    This gives the data a mean of 0 and variance of 1"""
    train = train.copy()
    test = test.copy()
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis = 0)
    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test

def unstandardize(B):
    """Unstandardizes coefficients by multiplying by sd of 
    y divided by sd of the x column associated with the coefficient. 
    Also unstandardizes the intercept by subtracting B* mean of X 
    from mean of Y"""
    intercept = B[0]
    coeffs = B[1:]
    for i in np.arange(len(coeffs)):
        coeffs[i] = coeffs[i] * (np.std(wine_data.iloc[:,-1].values)/np.std(wine_data.iloc[:,i].values))
    intercept = intercept - np.dot(np.mean(wine_data.iloc[:,0:-1].values,0)
                                   ,coeffs)
    return np.append(intercept,coeffs)

def VIF(train_df):
    """
    This function computes variance inflation factors (VIF)
    for a given feature matrix 

    VIF scores of 5 or higher indicate multicollinearity among the feature vectors

    inputs: X - a feature matrix, size m by n, type numpy ndarray 
    outputs: vif - array, size n-1
    """
    train_df = train_df.copy()
    del train_df['quality']
    X = train_df.values
    ncols = X.shape[1]
    vif = np.zeros(ncols)
    for i in np.arange(ncols):
        y_vif =  X[:,i]
        X_vif =  np.delete(X, i, 1)
        B = lr.ols(X_vif, y_vif, fit_intercept = False)
        Y_hat = np.dot(X_vif,B) 
        Y_bar = np.mean(y_vif)
        SSR = np.dot(np.transpose(Y_hat - Y_bar), Y_hat - Y_bar)
        SSTO = np.dot(np.transpose(y_vif-Y_bar),y_vif-Y_bar)
        R2 = SSR/SSTO
        vif[i] = 1/(1-R2)  
    if all(i < 5 for i in vif):
        print("Low Multicollinearity Detected (All VIF Scores Less Than Five")   
    else:
        print("High Multicollinearity Detected")
    return pd.DataFrame(vif, index = train_df.columns).T

def soft_thresholding_operator(z, gamma):
    if z > 0 and gamma < abs(z):
        return z - gamma
    elif z < 0 and gamma < abs(z):
        return z + gamma 
    elif gamma >= abs(z):
        return 0

def get_error(B, test_data):
	y_hat = np.dot(test_data[:,0:-1],B[1:]) + B[0]
	error = np.linalg.norm(test_data[:,-1]-y_hat)**2
	return error

def create_comparison_table(times, model, sklearn_model, test_data, features):
    model_error = get_error(np.ndarray.flatten(model), test_data)
    sk_error = get_error(np.append(sklearn_model.intercept_[0], 
                                         sklearn_model.coef_), test_data)
    row_model = np.append(np.append(times[0], model_error), model)
    row_sklearn = np.append([times[1], sk_error, 
                    sklearn_model.intercept_[0]],
                    sklearn_model.coef_)
    df = pd.DataFrame(np.vstack((row_model, row_sklearn)),
                         columns = np.append(['Runtime (s)','Error','Y-Intercept'], features))
    df.index = ['My Function', "Scikit-Learn's Function"]
    return df

def compare_ols(X_train, y_train, test_data, features):
    start = time.time()
    model = lr.ols(X_train, y_train)
    end = time.time()
    times = end-start

    start = time.time()
    sklearn_model = skols(fit_intercept= True).fit(X_train, y_train)
    end = time.time()
    times = np.append(times, end - start)
    return create_comparison_table(times, model, sklearn_model, test_data, features)


def compare_ridge(X_train, y_train, test_data, features, l):
    X_train_std, X_test_std = standardize(X_train, test_data[:, 0:-1])
    test_data_std = np.column_stack((X_test_std, test_data[:, -1]))
    start = time.time()
    model = lr.ridge(X_train_std, y_train, l)
    end = time.time()
    times = end-start

    start = time.time()
    sklearn_model = skridge(l, fit_intercept= True).fit(X_train_std, y_train)
    end = time.time()
    times = np.append(times, end - start)
    return create_comparison_table(times, model, sklearn_model, test_data_std, features)

def compare_lasso(X_train, y_train, test_data, features, l):
    X_train_std, X_test_std = standardize(X_train, test_data[:, 0:-1])
    test_data_std = np.column_stack((X_test_std, test_data[:, -1]))
    start = time.time()
    model = lr.lasso(X_train_std, y_train, l1=l)
    end = time.time()
    times = end-start

    start = time.time()
    sklearn_model = sklasso(l, fit_intercept= True).fit(X_train_std, y_train)
    end = time.time()
    times = np.append(times, end - start)
    return create_comparison_table(times, model, sklearn_model, test_data_std, features)

def compare_elastic_net(X_train, y_train, test_data, features, l1, l2):
    X_train_std, X_test_std = standardize(X_train, test_data[:, 0:-1])
    test_data_std = np.column_stack((X_test_std, test_data[:, -1]))
    start = time.time()
    model = lr.elastic_net(X_train_std, y_train, l1, l2)
    end = time.time()
    times = end-start

    start = time.time()
    sklearn_model = skelasticnet(alpha = l1 + l2, l1_ratio= l1/(l1+l2), fit_intercept= True).fit(X_train_std, y_train)
    end = time.time()
    times = np.append(times, end - start)
    return create_comparison_table(times, model, sklearn_model, test_data_std, features)

