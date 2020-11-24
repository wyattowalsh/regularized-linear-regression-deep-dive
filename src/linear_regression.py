import pandas as pd
import numpy as np
from math import *
from src import utilities as utils

def ols(df):
    """This function takes in a dataframe with x data and 
    y data as the last column.It then proceeds to return the 
    OLS estimates for the different parameters using the 
    closed form solution"""
    X = np.hstack((np.ones((len(df),1)),df[:,0:-1]))
    y = df[:,-1]
    B = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), (np.dot(np.transpose(X),y)))
    return B

def ridge_regression(df, index, num_parameters):
    """Augments data with a 1 column and a 
    diag(square root of lambda*I) then computes OLS"""
    u = np.geomspace(1e-2,1e5,num_parameters)[index]
    upper_half = np.append(np.ones((len(df),1)),
                           df[:,0:-1],1)
    lower_half = np.append(np.zeros((len(df.columns)-1,1)), 
                           np.sqrt(u)* 
                           np.identity(len(df.columns)-1),1)
    X = np.append(upper_half,lower_half,0)
    Y = np.append(df[:,-1],np.zeros(
        len(df.columns)-1),0)
    B = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),(np.dot(np.transpose(X),Y)))
    return B

def lasso_regression_covar(df,u,u_path = np.array([])):
    """Solves lasso regression. Since no closed form solution 
    exists we must use optimization to solve for the parameters. 
    Here we implement coordinate descent (CD) with warm starts 
    utilizing an equation for u_max so that we can create a log 
    spaced vector for u. Using the CD update we iterate until 
    the coeffiecients converge and stabilize."""
    
    df_std = utils.standardize(df)
    X = np.append(np.ones((len(df),1)),df_std.iloc[:,0:-1].values,1)
    Y = df.iloc[:,-1].values
    N = np.shape(X)[0]
    
    B_star = np.zeros((np.shape(X)[1]))
    
    u_max = max(list(abs(np.dot(np.transpose(X),Y))))/len(df)
    
    if u_path.size == 0:
        if u == 0:
            u_ = np.append(np.geomspace(u_max,u+1e-10,99),0)
        else: 
            u_ = np.geomspace(u_max,u,100)
    else:   
        u_ = u_path
        
    tot_coeff = np.zeros((len(u_),len(B_star)))  
    
    for i in np.arange(len(u_)):
        while True:
            B_s = B_star
            for j in np.arange(len(B_s)):
                k = np.where(B_s!= 0)[0]
                update = (1/N)*((np.dot(np.transpose(X[:,j]),Y) - 
                                 np.dot(np.dot(np.transpose(X[:,j]),X[:,k]),B_s[k])))+B_s[j]
                B_star[j] = (np.sign(update)*max(abs(update)-u_[i],0))
            tot_coeff[i,:] = B_star
            if np.all(np.absolute(np.subtract(B_s, B_star)) < 1e-3):
                break
    return B_star, tot_coeff

def elastic_net(df,u,alpha,u_path = np.array([])): 
    """Solves elastic net regression. Since no closed form solution exists 
    we must use optimization to solve for the parameters. Here we implement 
    coordinate descent (CD) with warm starts utilizing an equation for u_max 
    so that we can create a log spaced vector for u. Using the CD update we 
    iterate until the coeffiecients converge and stabilize."""
    dff = utils.standardize(df)
    X = np.append(np.ones((len(df),1)),dff.iloc[:,0:-1].values,1)
    Y = df.iloc[:,-1].values
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

