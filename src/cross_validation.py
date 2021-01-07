from src import utilities as utils
from src import linear_regression as lr
from itertools import product
import numpy as np
import pandas as pd


def get_folds(X, y, k=5):
    """This function uses K-Fold C.V. to choose tuning parameters for 
    the different regression techniques. It not only returns the 
    best parameters, but also the parameters tested, the coeefficients 
    associated with them, the minimum error, and the error for each 
    parameter for the last fold."""
    df = np.hstack((X, y))
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    sets = np.array_split(indices, k)
    fn = lambda index: df[sets[index], :]
    return [fn(i) for i in np.arange(5)]


def ridge(X, y, num_lambdas=100, k=5):
    '''This function performs K-Fold Cross-Validation for Ridge Regression. 
    This is accomplished through numpy operations on a numpy matrix input over 
    a given number of tuning parameters (n) and number of folds (k)'''
    tuning_params = np.geomspace(1e-6, 50000, num_lambdas)
    B_list = []
    errors = []
    folds = get_folds(X, y, k)
    X = [folds[i][:, 0:-1] for i in np.arange(k)]
    y = [folds[i][:, [-1]] for i in np.arange(k)]
    fold_index = np.arange(k)
    for i in fold_index:
        X_test, y_test = X[i], y[i]
        X_train = np.vstack([X[i] for i in np.delete(fold_index, i)])
        y_train = np.vstack([y[i] for i in np.delete(fold_index, i)])
        X_train_std, X_test_std = utils.standardize(X_train, X_test)
        B = [
            lr.ridge(X_train_std, y_train, tuning_params[i])
            for i in np.arange(num_lambdas)
        ]
        error = [
            utils.get_error(B[i], np.hstack((X_test_std, y_test)))
            for i in np.arange(num_lambdas)
        ]
        B_list = B_list + [B]
        errors = errors + [error]
    fold_means = np.mean(np.array(errors), 0)
    optimal_tune_index = fold_means.argmin()
    tune_star = tuning_params[optimal_tune_index]
    return ([
        errors, fold_means, fold_means[optimal_tune_index], B_list,
        tuning_params, tune_star
    ])


def lasso(X_train, y_train, num_lambdas=100, k=5):
    '''This function performs K-Fold Cross-Validation for Ridge Regression. 
    This is accomplished through numpy operations on a numpy matrix input over 
    a given number of tuning parameters (n) and number of folds (k)'''
    tuning_params = np.geomspace(1e-6, 1, num_lambdas)
    B_list = []
    errors = []
    folds = get_folds(X_train, y_train, k)
    X = [folds[i][:, 0:-1] for i in np.arange(k)]
    y = [folds[i][:, [-1]] for i in np.arange(k)]
    fold_index = np.arange(k)
    for i in fold_index:
        X_test, y_test = X[i], y[i]
        X_train = np.vstack([X[j] for j in np.delete(fold_index, i)])
        y_train = np.vstack([y[j] for j in np.delete(fold_index, i)])
        X_train_std, X_test_std = utils.standardize(X_train, X_test)
        B = [lr.lasso(X_train_std, y_train, l1=l) for l in tuning_params]
        error = [
            utils.get_error(B[j], np.hstack((X_test_std, y_test)))
            for j in np.arange(num_lambdas)
        ]
        B_list = B_list + [B]
        errors = errors + [error]
    fold_means = np.mean(np.array(errors), 0)
    optimal_tune_index = fold_means.argmin()
    tune_star = tuning_params[optimal_tune_index]
    return [
        errors, fold_means, fold_means[optimal_tune_index], B_list,
        tuning_params, tune_star
    ]


def elastic_net(X_train, y_train, num_lambdas=50, k=5):
    l = np.geomspace(1e-6, 50000, num_lambdas)
    alpha = np.linspace(1e-6, 1, num_lambdas)
    tuning_params = list(product(l, alpha))

    B_list = []
    errors = []
    folds = get_folds(X_train, y_train, k)
    X = [folds[i][:, 0:-1] for i in np.arange(k)]
    y = [folds[i][:, [-1]] for i in np.arange(k)]
    fold_index = np.arange(k)
    for i in fold_index:
        X_test, y_test = X[i], y[i]
        X_train = np.vstack([X[j] for j in np.delete(fold_index, i)])
        y_train = np.vstack([y[j] for j in np.delete(fold_index, i)])
        X_train_std, X_test_std = utils.standardize(X_train, X_test)
        B = [
            lr.elastic_net(X_train_std, y_train, tuning_param[0], tuning_param[1])
            for tuning_param in tuning_params
        ]
        error = [
            utils.get_error(B[j], np.hstack((X_test_std, y_test)))
            for j in np.arange(num_lambdas)
        ]
        B_list = B_list + [B]
        errors = errors + [error]
    fold_means = np.mean(np.array(errors), 0)
    optimal_tune_index = fold_means.argmin()
    tune_star = tuning_params[optimal_tune_index]
    return [
        errors, fold_means, fold_means[optimal_tune_index], tuning_params,
        tune_star
    ]