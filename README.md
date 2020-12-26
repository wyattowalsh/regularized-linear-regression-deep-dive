# Regularized Linear Regression From Scratch 

Link to Binder cloud-hosted notebook: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wyattowalsh/regularized-regression-from-scratch/HEAD?filepath=nb.ipynb)

--- 

## Project Overview

Here I implement Ordinary Least Squares (OLS) regression, Ridge regression, Lasso regression, and Elastic Net regression from scratch using only the NumPy and math Python libraries. 

For the cases of OLS and Ridge regression, model estimators are derived analytically and their uniqueness is proven.

In the cases of the Lasso and the Elastic Net, a closed-form solution cannot be analytically derived due to the piece-wise nature of the L<sub>1</sub> penalty. To overcome this matter, the pathwise Coordinate Descent algorithm is implemented and utilized. 

These regression algorithms are applied to a wine quality prediction dataset. 

Functions for creating a train-test split as well as feature standardizization are implemented. Furthermore, the dataset is analyzed for multicollinearity among features. For the models containing hyperparameters (all the regularization techniques), K-Fold cross-validation is also implemented. Runtime, error, and model parameter comparisons with the associated Scikit-Learn algorithms are given, with the results of the author's implementations surpassing the results of Scikit-Learn for every algorithm. Visualizations of model parameter values across a range of hyperparameter values are generated. Finally, model accuracies are compared and a recommendation is given for which model to use for this case. 

To gain the best sense of the project I recommend viewing  `nb.ipynb` either locally or through Binder and then looking at the associated model code within the `src` directory of this repository if interested. 
