# This is a script of utility functions utilized throughout the project

# Necessary imports
from math import *
import numpy as np
from src import linear_regression as lr
import time

import pandas as pd

from sklearn.linear_model import LinearRegression as skols
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

def VIF(X):
    """
    This function computes variance inflation factors (VIF)
    for a given feature matrix 

    VIF scores of 5 or higher indicate multicollinearity among the feature vectors

    inputs: X - a feature matrix, size m by n, type numpy ndarray 
    outputs: vif - array, size n-1
    """
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
        return "Low Multicollinearity!",vif
    else:
         return "High Multicollinearity",vif 

def get_error(B, test_data):
	y_hat = np.dot(test_data[:,0:-1],B[1:])+B[0]
	error = np.linalg.norm(test_data[:,-1]-y_hat)**2
	return(error)

def compare(type, X_train, y_train, X_test, y_test, features):
    test_vals = np.hstack((X_test, y_test))
    if type == 'ols' or type == 'OLS':
        start = time.time()
        ols = lr.ols(X_train, y_train)
        end = time.time()
        times_OLS = end-start
        ols_error = get_error(np.ndarray.flatten(ols), test_vals)

        start = time.time()
        OLS_fitted_sklearn = skols(fit_intercept= True).fit(X_train, y_train)
        end = time.time()
        times_OLS = np.append(times_OLS, end - start)
        sk_error = get_error(np.append(OLS_fitted_sklearn.intercept_[0], 
                                             OLS_fitted_sklearn.coef_), test_vals)

        print('Are the error values close?', " ", isclose(ols_error,sk_error))

        ols_row = np.append(np.append(times_OLS[0], ols_error), ols)
        OLS_sklearn_row = np.append([times_OLS[1], sk_error, 
                                    OLS_fitted_sklearn.intercept_[0]],
                                    OLS_fitted_sklearn.coef_)
        OLS_df = pd.DataFrame(np.vstack((ols_row, OLS_sklearn_row)),
                             columns = np.append(['Runtime (s)','Error','Y-Intercept'], features))
        OLS_df.index = ['My Function', "Scikit-Learn's Function"]
        return(OLS_df)





