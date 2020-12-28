import pandas as pd
import numpy as np
import jax.numpy as jnp 
from math import *
from src import utilities as utils


def ols(X, y, fit_intercept = True):
    """This function takes in a dataframe with x data and 
    y data as the last column.It then proceeds to return the 
    OLS estimates for the different parameters using the 
    closed form solution"""
    m,n = np.shape(X)
    if fit_intercept:
        X = np.hstack((np.ones((m,1)), X))
    return np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))
    
def ridge(X, y, l2):
    """Augments data with a 1 column and a 
    diag(square root of lambda*I) then computes OLS"""
    m, n = np.shape(X)
    upper_half = np.hstack((np.ones((m,1)), X))
    lower = np.zeros((n,n))
    np.fill_diagonal(lower, np.sqrt(l2))
    lower_half = np.hstack((np.zeros((n,1)), lower))
    X = np.vstack((upper_half,lower_half))
    y = np.append(y,np.zeros(n))
    return np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))

def lasso(X, y, l1, tol=1e-6, path_length=100, return_path=False):
    X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    l_max = max(list(abs(np.dot(np.transpose(X[:, 1:]),y))))/m
    if l1 >= l_max:
        return np.append(np.mean(y), np.zeros((n-1)))
    l_path = np.geomspace(l_max, l1, path_length)
    coeffiecients = np.zeros((len(l_path), n))
    for i in range(len(l_path)):
        while True:
            B_s = B_star
            for j in range(n):
                k = np.where(B_s!= 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(abs(update)-l_path[i],0))
            if np.all(abs(B_s - B_star) < tol):
                coeffiecients[i, :] = B_star
                break
    if return_path:
        return [B_star, l_path, coeffiecients]
    else:
        return B_star


def elastic_net(X, y, l1, l2, l1_path = np.array([]), tol = 1e-4): 
    """Solves elastic net regression. Since no closed form solution exists 
    we must use optimization to solve for the parameters. Here we implement 
    coordinate descent (CD) with warm starts utilizing an equation for u_max 
    so that we can create a log spaced vector for u. Using the CD update we 
    iterate until the coeffiecients converge and stabilize."""

    X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    path_length = 100
    if l2 == 0:
        l2 = 1e-15
    l_max = max(list(abs(np.dot(np.transpose(X),y))))/m/l2
    
    if l1_path.size == 0:
        if l1 == 0:
            l1_path = np.append(np.geomspace(l_max, l_max * 1e-4, path_length - 1),0)
        else: 
            l1_path = np.geomspace(l_max, l1, path_length)
    
    for i in range(path_length):
        while True:
            B_s = B_star
            for j in range(n):
                k = np.where(B_s!= 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(abs(update)-l1_path[i] * l2,0))/ (1 + (l1_path[i] * (1 - l2)))
            if np.all(abs(B_s - B_star) < tol):
                break
    return B_star

