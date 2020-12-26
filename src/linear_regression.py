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
    
def ridge(X, y, l):
    """Augments data with a 1 column and a 
    diag(square root of lambda*I) then computes OLS"""
    m, n = np.shape(X)
    upper_half = np.hstack((np.ones((m,1)), X))
    lower = np.zeros((n,n))
    np.fill_diagonal(lower, np.sqrt(l))
    lower_half = np.hstack((np.ones((n,1)), lower))
    X = np.vstack((upper_half,lower_half))
    y = np.append(y,np.zeros(n))
    return np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))

def lasso(X, y, l=0, return_path=False, l_path = np.array([]), tol=1e-3):
    """Solves lasso regression. Since no closed form solution 
    exists we must use optimization to solve for the parameters. 
    Here we implement coordinate descent (CD) with warm starts 
    utilizing an equation for u_max so that we can create a log 
    spaced vector for u. Using the CD update we iterate until 
    the coefficients converge and stabilize."""
    
    m, n = np.shape(X)
    B_star = np.zeros((n))
    
    l_max = max(list(abs(np.dot(np.transpose(X),y))))/len(y)
    
    if l_path.size == 0:
        if l == 0:
            l_ = np.append(np.geomspace(l_max,l+1e-10,99),0)
        else: 
            l_ = np.geomspace(l_max,l,100)
    else:   
        l_ = l_path
        
    tot_coeff = np.zeros((len(l_),len(B_star)))  
    
    for i in np.arange(len(l_)):
        while True:
            B_s = B_star
            for j in np.arange(len(B_s)):
                k = np.where(B_s!= 0)[0]
                update = (1/m)*((np.dot(np.transpose(X[:,j]),y) - \
                                np.dot(np.dot(np.transpose(X[:,j]),X[:,k]),B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update)*max(abs(update)-l_[i],0))
            tot_coeff[i,:] = B_star
            if np.all(np.absolute(np.subtract(B_s, B_star)) < tol):
                break
    B_star = np.append(np.mean(y), B_star)
    tot_coeff = np.column_stack((np.ones((len(l_),1)) * np.mean(y), tot_coeff))
    if return_path:
        return B_star, tot_coeff
    else:
        return B_star


def elastic_net(df,u,alpha,u_path = np.array([])): 
    """Solves elastic net regression. Since no closed form solution exists 
    we must use optimization to solve for the parameters. Here we implement 
    coordinate descent (CD) with warm starts utilizing an equation for u_max 
    so that we can create a log spaced vector for u. Using the CD update we 
    iterate until the coeffiecients converge and stabilize."""
    X = np.append(np.ones((np.shape(df)[0],1)),df[:,0:-1],1)
    Y = df[:,-1].values
    N = len(df)
    if alpha == 0:
        alpha = 1e-15
    
    u_max = max(list(abs(np.dot(np.transpose(X),Y))))/len(df)/alpha
    
    if u_path.size == 0:
        if u == 0:
            u_ = np.append(np.geomspace(u_max,u+1e-10,99),0)
        else: 
            u_ = np.geomspace(u_max,u,100)
    else:   
        u_ = u_path
        
    B_star = np.zeros(np.shape(X)[1])
    
    tot_coeff = np.zeros((len(u_),len(B_star)))  
    
    for i in np.arange(len(u_)):
        while True:
            B_s = B_star
            for j in np.arange(len(B_s)):
                k = np.where(B_s!= 0)[0]
                update = (1/N)*((np.dot(np.transpose(X[:,j]),Y) - 
                                 np.dot(np.dot(np.transpose(X[:,j]), X[:,k]),B_s[k]))) + B_s[j]
                B_star[j] = (np.sign(update)*
                             max(abs(update)-(alpha*u_[i]),0)) / (1+(u_[i]*(1-alpha)))
            tot_coeff[i,:] = B_star
            if np.all(np.absolute(np.subtract(B_s, B_star)) < 1e-6):
                break
    return B_star,tot_coeff,u_

