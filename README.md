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

Sources: 
- [Regularization Paths for Generalized Linear Models via Coordinate Descent Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2929880/)
- [Fast Regularization Paths via Coordinate Descent Talk](https://web.stanford.edu/~hastie/TALKS/glmnet.pdf)
- [UC Berkeley Introduction to Machine Learning (EECS 189) Note on Linear Regression](https://www.eecs189.org/static/notes/n2.pdf)
- [UC Berkeley Introduction to Machine Learning (EECS 189) Note on Optimization](https://www.eecs189.org/static/notes/n12.pdf)
- [UC Berkeley Engineering Statistics, Quality Control, and Forecasting (IEOR 165) Note on Linear Regression](http://courses.ieor.berkeley.edu/ieor165/lecture_notes/ieor165_lec3.pdf)
- [UC Berkeley Engineering Statistics, Quality Control, and Forecasting (IEOR 165) Note on Bias-Variance Tradeoff](http://courses.ieor.berkeley.edu/ieor165/lecture_notes/ieor165_lec7.pdf)
- [UC Berkeley Engineering Statistics, Quality Control, and Forecasting (IEOR 165) Note on Regularization](http://courses.ieor.berkeley.edu/ieor165/lecture_notes/ieor165_lec8.pdf)
- [Stanford Statistics 305 Lecture on Ridge Regression and the Lasso](http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf)
- [The Elements of Statistical Learning Book](https://web.stanford.edu/~hastie/ElemStatLearn//index.html)
- [Coordinate Descent Lecture from Carnegie Mellon University](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/25-coord-desc.pdf)
