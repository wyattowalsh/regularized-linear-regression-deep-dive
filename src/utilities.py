'''Project utility functions: 
train/test splitting, data standardization, variance inflation factor calculation
model error calculation, and Scikit-Learn comparisons'''

import time
import numpy as np
import pandas as pd
from math import floor
from src import linear_regression as lr
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 


def test_train_split(df, proportion_train):
    """Shuffles and splits given dataframe

    Params:
        df - Pandas dataframe to be split
        proportion_train - Ratio between 0 and 1

    Returns:
        Tuple of training dataframe and testing dataframe split from input dataframe
    """
    shuffled = df.sample(frac=1, random_state=18).reset_index(drop=True)
    split = floor(len(shuffled) * proportion_train)
    train = shuffled.iloc[:split, :]
    test = shuffled.iloc[split:, :].reset_index(drop=True)
    return train, test


def standardize(train, test):
    """Ordinary Least Squares (OLS) Regression model with intercept term.
    Fits an OLS regression model using the closed-form OLS estimator equation.
    Intercept term is included via design matrix augmentation.

    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response

    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    train = train.copy()
    test = test.copy()
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis=0)
    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test


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
        y_vif = X[:, i]
        X_vif = np.delete(X, i, 1)
        B = lr.ols(X_vif, y_vif, fit_intercept=False)
        Y_hat = np.dot(X_vif, B)
        Y_bar = np.mean(y_vif)
        SSR = np.dot(np.transpose(Y_hat - Y_bar), Y_hat - Y_bar)
        SSTO = np.dot(np.transpose(y_vif - Y_bar), y_vif - Y_bar)
        R2 = SSR / SSTO
        vif[i] = 1 / (1 - R2)
    if all(i < 5 for i in vif):
        print("Low Multicollinearity Detected (All VIF Scores Less Than Five")
    else:
        print("High Multicollinearity Detected")
    return pd.DataFrame(vif, index=train_df.columns).T


def get_error(B, test_data):
    y_hat = np.dot(test_data[:, 0:-1], B[1:]) + B[0]
    error = np.linalg.norm(test_data[:, -1] - y_hat)**2
    return error


def create_comparison_table(times, model, sklearn_model, test_data, features):
    model_error = get_error(np.ndarray.flatten(model), test_data)
    sk_error = get_error(
        np.append(sklearn_model.intercept_[0], sklearn_model.coef_), test_data)
    row_model = np.append(np.append(times[0], model_error), model)
    row_sklearn = np.append([times[1], sk_error, sklearn_model.intercept_[0]],
                            sklearn_model.coef_)
    df = pd.DataFrame(np.vstack((row_model, row_sklearn)),
                      columns=np.append(
                          ['Runtime (s)', 'Error', 'Y-Intercept'], features))
    df.index = ['My Function', "Scikit-Learn's Function"]
    return df

def compare(X_train, y_train, test_data, features, model, l1=None, l2=None):
    if model == lr.ols:
        start = time.time()
        model = model(X_train, y_train)
        end = time.time()
        times = end - start
        start = time.time()
        sklearn_model = LinearRegression(fit_intercept=True).fit(X_train, y_train)
        end = time.time()
    elif model == lr.ridge:
        X_train_std, X_test_std = standardize(X_train, test_data[:, 0:-1])
        test_data = np.column_stack((X_test_std, test_data[:, -1]))
        start = time.time()
        model = model(X_train_std, y_train, l2=l2)
        end = time.time()
        times = end - start
        start = time.time()
        sklearn_model = Ridge(l2, fit_intercept=True).fit(X_train_std, y_train)
        end = time.time()
    elif model == lr.lasso:
        X_train_std, X_test_std = standardize(X_train, test_data[:, 0:-1])
        test_data = np.column_stack((X_test_std, test_data[:, -1]))
        start = time.time()
        model = model(X_train_std, y_train, l1=l1)
        end = time.time()
        times = end - start
        start = time.time()
        sklearn_model = Lasso(l1, fit_intercept=True).fit(X_train_std, y_train)
        end = time.time()
    elif model == lr.elastic_net:
        X_train_std, X_test_std = standardize(X_train, test_data[:, 0:-1])
        test_data = np.column_stack((X_test_std, test_data[:, -1]))
        start = time.time()
        model = model(X_train_std, y_train, l1=l1, l2=l2)
        end = time.time()
        times = end - start
        start = time.time()
        sklearn_model = ElasticNet(alpha=l1, l1_ratio=l2, fit_intercept=True).fit(X_train_std, y_train)
        end = time.time()
    
    times = np.append(times, end - start)
    return create_comparison_table(times, model, sklearn_model, test_data, features)