import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src import utilities as utils
from src import linear_regression as lr


def scatter_matrix(train):
    sns.set_theme(context='notebook',
                  style='white',
                  font='sans-serif',
                  font_scale=1.8)
    plot = sns.pairplot(train, hue='quality', palette='icefire')
    return plot

def correlation_heatmap(train_df):
    train_df = train_df.copy()
    del train_df["quality"]
    sns.set_theme(context='notebook',
                  style='white',
                  font='sans-serif',
                  font_scale=1.8)
    plot = sns.heatmap(
        train_df.corr(),
        annot=True).set_title("Numerical Feature Correlation Heatmap")
    return plot


def ridge_plot(X_train, y_train, num_lambdas, features):
    sns.set_theme(context='notebook',
                  style='whitegrid',
                  font='sans-serif',
                  font_scale=1.8)
    X_train_std = utils.standardize(X_train, [1])[0]
    tuning_params = np.geomspace(1e-6, 35000, num_lambdas)
    B = [lr.ridge(X_train_std, y_train, l2) for l2 in tuning_params]
    df = pd.DataFrame(np.column_stack((tuning_params, B)),
                      columns=['L2 Penalty', 'Intercept', *features])
    del df['Intercept']
    df = pd.melt(df,
                 'L2 Penalty').rename(columns={'value': 'Coefficient Value'})
    plot = sns.lineplot(data= df, x = 'L2 Penalty', y = 'Coefficient Value', \
         hue = 'variable', palette= "icefire", linewidth = 3.5).\
         set_title('Ridge Coefficient Paths')
    return plot


def lasso_plot(X_train, y_train, num_lambdas, features):
    sns.set_theme(context='notebook',
                  style='whitegrid',
                  font='sans-serif',
                  font_scale=2)
    X_train_std = utils.standardize(X_train, [1])[0]
    lasso = lr.lasso(X_train_std,
                     y_train,
                     l1=1e-9,
                     path_length=num_lambdas,
                     return_path=True)
    df = pd.DataFrame(np.column_stack((lasso[1], lasso[2])),
                      columns=['L1 Penalty', 'Intercept', *features])
    del df['Intercept']
    df = pd.melt(df,
                 'L1 Penalty').rename(columns={'value': 'Coefficient Value'})
    plot = sns.lineplot(data= df, x = 'L1 Penalty', y = 'Coefficient Value', \
         hue = 'variable', palette= "icefire", linewidth = 3.5).\
         set_title('Lasso Coefficient Paths')
    return plot