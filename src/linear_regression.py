"""Ordinary Least Squares, Ridge, Lasso, and Elastic Net regression modeling functions"""

import numpy as np


def ols(X, y, fit_intercept=True):
    """Ordinary Least Squares (OLS) Regression model with intercept term.
    Fits an OLS regression model using the closed-form OLS estimator equation.
    Intercept term is included via design matrix augmentation.

    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        fit_intercept - Boolean indicating whether to include an intercept term

    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    m, n = np.shape(X)
    if fit_intercept:
        X = np.hstack((np.ones((m, 1)), X))
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


def ridge(X, y, l2):
    """Ridge Regression model with intercept term.
    L2 penalty and intercept term included via design matrix augmentation.
    This augmentation allows for the OLS estimator to be used for fitting.

    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        l2 - L2 penalty tuning parameter (positive scalar) 

    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    m, n = np.shape(X)
    upper_half = np.hstack((np.ones((m, 1)), X))
    lower = np.zeros((n, n))
    np.fill_diagonal(lower, np.sqrt(l2))
    lower_half = np.hstack((np.zeros((n, 1)), lower))
    X = np.vstack((upper_half, lower_half))
    y = np.append(y, np.zeros(n))
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


def lasso(X, y, l1, tol=1e-6, path_length=100, return_path=False):
    """The Lasso Regression model with intercept term.
    Intercept term included via design matrix augmentation.
    Pathwise coordinate descent with co-variance updates is applied.
    Path from max value of the L1 tuning parameter to input tuning parameter value.
    Features must be standardized (centered and scaled to unit variance)

    Params:
        X - NumPy matrix, size (N, p), of standardized numerical predictors
        y - NumPy array, length N, of numerical response
        l1 - L1 penalty tuning parameter (positive scalar)
        tol - Coordinate Descent convergence tolerance (exited if change < tol)
        path_length - Number of tuning parameter values to include in path (positive integer)
        return_path - Boolean indicating whether model coefficients along path should be returned

    Returns:
        if return_path == False:
            NumPy array, length p + 1, of fitted model coefficients
        if return_path == True:
            List, length 3, of last fitted model coefficients, tuning parameter path and coefficient values
    """
    X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    l_max = max(list(abs(np.dot(np.transpose(X[:, 1:]), y)))) / m
    # At or above l_max, all coefficients (except intercept) will be brought to 0
    if l1 >= l_max:
        return np.append(np.mean(y), np.zeros((n - 1)))
    l_path = np.geomspace(l_max, l1, path_length)
    coeffiecients = np.zeros((len(l_path), n))
    for i in range(len(l_path)):
        while True:
            B_s = B_star
            for j in range(n):
                k = np.where(B_s != 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(abs(update) - l_path[i], 0))
            if np.all(abs(B_s - B_star) < tol):
                coeffiecients[i, :] = B_star
                break
    if return_path:
        return [B_star, l_path, coeffiecients]
    else:
        return B_star


def elastic_net(X, y, l, alpha, tol=1e-4, path_length=100, return_path=False):
    """The Elastic Net Regression model with intercept term.
    Intercept term included via design matrix augmentation.
    Pathwise coordinate descent with co-variance updates is applied.
    Path from max value of the L1 tuning parameter to input tuning parameter value.
    Features must be standardized (centered and scaled to unit variance)

    Params:
        X - NumPy matrix, size (N, p), of standardized numerical predictors
        y - NumPy array, length N, of numerical response
        l1 - L1 penalty tuning parameter (positive scalar)
        l2 - L2 penalty tuning parameter (positive scalar)
        tol - Coordinate Descent convergence tolerance (exited if change < tol)
        path_length - Number of tuning parameter values to include in path (positive integer)

    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    if alpha == 0:
        alpha = 1e-15
    l_max = max(list(abs(np.dot(np.transpose(X), y)))) / m / alpha
    if l >= l_max:
        return np.append(np.mean(y), np.zeros((n - 1)))
    l_path = np.geomspace(l_max, l, path_length)
    for i in range(path_length):
        while True:
            B_s = B_star
            for j in range(n):
                k = np.where(B_s != 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(
                    abs(update) - l_path[i] * alpha, 0)) / (1 + (l_path[i] * (1 - alpha)))
            if np.all(abs(B_s - B_star) < tol):
                break
    return B_star