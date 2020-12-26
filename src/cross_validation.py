from src import utilities as utils
from src import linear_regression as lr

import numpy as np
import pandas as pd

def get_folds(X, y, k=5):
    """This function uses K-Fold C.V. to choose tuning parameters for 
    the different regression techniques. It not only returns the 
    best parameters, but also the parameters tested, the coeefficients 
    associated with them, the minimum error, and the error for each 
    parameter for the last fold."""
    df = np.hstack((X,y))
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    sets = np.array_split(indices, k)
    fn = lambda index: df[sets[index],:]
    return [fn(i) for i in np.arange(5)]

def ridge(X, y, num_lambdas=100, k=5):
    '''This function performs K-Fold Cross-Validation for Ridge Regression. 
    This is accomplished through numpy operations on a numpy matrix input over 
    a given number of tuning parameters (n) and number of folds (k)'''
    tuning_params =  np.flip(np.geomspace(10000, 1e-10, num_lambdas))
    B_list = []
    errors = []
    folds = get_folds(X, y, k)
    X = [folds[i][:,0:-1] for i in np.arange(k)]
    y = [folds[i][:, [-1]] for i in np.arange(k)]
    fold_index = np.arange(k)
    for i in fold_index:
        X_test, y_test = X[i], y[i]
        X_train = np.vstack([X[i] for i in np.delete(fold_index, i)])
        y_train = np.vstack([y[i] for i in np.delete(fold_index, i)])
        X_train_std, X_test_std = utils.standardize(X_train, X_test)
        B = [lr.ridge(X_train_std, y_train, tuning_params[i]) for i in np.arange(num_lambdas)]
        error = [utils.get_error(B[i], np.hstack((X_test_std, y_test))) for i in np.arange(num_lambdas)]
        B_list = B_list + [B]
        errors = errors + [error]
    fold_means = np.mean(np.array(errors),0)
    optimal_tune_index = fold_means.argmin()
    tune_star = tuning_params[optimal_tune_index]
    B_star = lr.ridge(X_train, y_train, tune_star)
    return([errors, fold_means, fold_means[optimal_tune_index], B_star, tuning_params, tune_star])

def lasso(X, y, num_lambdas=100, k=5):
    '''This function performs K-Fold Cross-Validation for Ridge Regression. 
    This is accomplished through numpy operations on a numpy matrix input over 
    a given number of tuning parameters (n) and number of folds (k)'''
    X_std = utils.standardize(X, 0)[0]
    tuning_params_max = max(list(abs(np.dot(np.transpose(X_std),y)))) / (np.shape(X_std)[1])
    tuning_params = np.geomspace(tuning_params_max,tuning_params_max*0.0001, num_lambdas)
    B_list = []
    errors = []
    folds = get_folds(X, y, k)
    X = [folds[i][:,0:-1] for i in np.arange(k)]
    y = [folds[i][:, [-1]] for i in np.arange(k)]
    fold_index = np.arange(k)
    for i in fold_index:
        X_test, y_test = X[i], y[i]
        X_train = np.vstack([X[i] for i in np.delete(fold_index, i)])
        y_train = np.vstack([y[i] for i in np.delete(fold_index, i)])
        X_train_std, X_test_std = utils.standardize(X_train, X_test)
        B = [lr.lasso(X_train_std, y_train, l) for l in tuning_params] 
        error = [utils.get_error(B[i], np.hstack((X_test_std, y_test))) for i in np.arange(num_lambdas)]
        B_list = B_list + [B]
        errors = errors + [error]
    fold_means = np.mean(np.array(errors),0)
    optimal_tune_index = fold_means.argmin()
    tune_star = tuning_params[optimal_tune_index]
    B_star = lr.lasso(X_train, y_train, tune_star)
    return([errors, fold_means, fold_means[optimal_tune_index], B_star, tuning_params, tune_star])



def cross_validation_error(df, regression_fn, n, k=5, df_test = None):
    if regression_fn == regress.ridge:
        parameter_tuning = np.geomspace(1e-2,1e5, n)
    elif regression_fn == regress.lasso: 
        pass
    elif regression_fn == regress.elastic_net:
        pass
    else:
        parameter_tuning = None
    sets, indices = cross_validation_indices(df, k)
    error = np.zeros((n,k))
    B = np.zeros((n, k, len(df.columns)))
    for i in np.arange(k):
        for j in np.arange(n):
            train_indices = [p for p in indices if p not in sets[i]]
            train_data = df.values[train_indices,:]
            test_data = df.values[sets[i],:]
            if regression_fn == regress.ridge:
                B[j, i, :] = regression_fn(train_data, parameter_tuning[j])
            elif regression_fn == regress.lasso: 
                train_data, test_data = utils.standardize(train_data, test_data)
                X = train_data[:,0:-1]
                y = train_data[:,-1]
                parameter_tuning_max = max(list(abs(np.dot(np.transpose(X),y)))) / (np.shape(train_data)[1])
                parameter_tuning = np.geomspace(parameter_tuning_max,parameter_tuning_max*0.0001,n)
                B[j,i,:], coeffs = regression_fn(train_data, 1, parameter_tuning)
            elif regression_fn == regress.elastic_net:
                pass
            else:
                B[j, i, :] = regression_fn(train_data) 
            y_hat = np.dot(test_data[:,0:-1],B[j,i, 1:])+B[j,i,0]
            error[j,i] = np.linalg.norm(test_data[:,-1]-y_hat)**2
    error_mean = sum(error)/k
    z = np.unravel_index(error.argmin(), error.shape)
    try: 
        parameter_tuning_star = parameter_tuning[z[0]]
    except:
        parameter_tuning_star = None
    try:
        parameter_model = B[z[0], z[1], :]
    except:
        parameter_model = B
    to_return = dict(zip(['Error Mean', 'Errors', 'Model Parameters', 'Optimal Fitted Model Parameters', 
                         'Tuning Parameters', 'Optimal Tuning Parameter'], 
                         [error_mean, error, B, parameter_model, parameter_tuning, parameter_tuning_star]))
    return to_return


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