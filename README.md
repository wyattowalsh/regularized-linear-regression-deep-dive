<h1 align="center"> Regularized Linear Regression Deep Dive: <br> Application to Wine Quality Regression Dataset </h1>

This project consists of a deep dive on multiple linear regression (OLS) and its regularized variants (Ridge, the Lasso, and the Elastic Net) as well as Python implementations for exploratory data analysis, K-Fold cross-validation and modeling functions as applied to regression of a wine quality dataset. This examination applies optimization theory to either derive model estimator (for OLS and Ridge) or derive the update rule for Pathwise Coordinate Descent (the discrete optimization algorithm chosen and implemented to solve the Lasso and the Elastic Net). These derivations have accompanying Python implementations, which are leveraged to predict wine quality ratings within a supervised learning context.

<p align="center">
  Interact with the project notebook in your web browser using the <i>Binder</i> service  
<a href=https://mybinder.org/v2/gh/wyattowalsh/regularized-linear-regression-deep-dive/HEAD?filepath=nb.ipynb> <img src=https://mybinder.org/badge_logo.svg></a>
 <br><br>
</p>

<p align="center">
    <a href=#explanation-of-repository-contents>Explanation of Repository Contents</a>  ·
    <a href=#project-overview> Project Overview</a>  ·
    <a href=#installation-instructions> Installation Instructions</a> 
  <br><br>
</p>

---

## Explanation of Repository Contents 

- `data/` - contains the project's wine quality dataset 
- `src/` - holds all the project source code
- `nb.ipynb`  - project notebook
- `environment.yml` - Conda virtual environment reproduction file

---

## Project Overview  

Here I implement Ordinary Least Squares (OLS) regression, Ridge regression, Lasso regression, and Elastic Net regression from scratch using only basic Python libraries

For the cases of OLS and Ridge regression, model estimators are derived analytically and their uniqueness is proven.

In the cases of the Lasso and the Elastic Net, a closed-form solution cannot be analytically derived due to the piece-wise nature of the L<sub>1</sub> penalty. To overcome this matter, the Pathwise Coordinate Descent algorithm is implemented and utilized. 

These regression algorithms are applied to a wine quality prediction dataset obtained through a class final project at UC Berkeley. 

Functions for creating a train-test split as well as feature standardizization are implemented. Furthermore, the dataset is analyzed for multicollinearity among features. For the models containing hyperparameters (all the regularization techniques), K-Fold cross-validation is also implemented. Runtime, error, and model parameter comparisons with the associated Scikit-Learn algorithms are given, with the results of the author's implementations surpassing the results of Scikit-Learn for every algorithm. Visualizations of model parameter values across a range of hyperparameter values are generated. Finally, model accuracies are compared and a recommendation is given for which model to use for this case. 

To gain the best sense of the project I recommend viewing  `nb.ipynb` either locally or through Binder and then looking at the associated model code within the `src` directory of this repository if interested. 

--- 

## Installation Instructions 

`environment.yml`  can be found in the repository's root directory for your version of interest and used to install necessary project dependencies. If able to successfully configure your computing environment, then launch Jupyter Notebook from your command prompt and navigate to `nb.ipynb`. If unable to successfully configure your computing environment refer to the sections below to install necessary system tools and package dependencies. The following sections may be cross-platform compatibile in several places, however is geared towards macOS<sup>[1](#footnote1)</sup>.

### Do you have the Conda system installed?

Open a command prompt (i.e. *Terminal*) and run: `conda info`.

This should display related information pertaining to your system's installation of Conda. If this is the case, you should be able to skip to the section regarding virtual environment creation (updating to the latest version of Conda could prove helpful though: `conda update conda`).

If this resulted in an error, then install Conda with the following section. 

### Install Conda

There are a few options here. To do a general full installation check out the [Anaconda Download Page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). However, the author strongly recommends the use of Miniconda since it retains necessary functionality while keeping resource use low; [Comparison with Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda) and [Miniconda Download Page](https://docs.conda.io/en/latest/miniconda.html). 

Windows users: please refer to the above links to install some variation of Conda. Once installed, proceed to the instructions for creating and configuring virtual environments [found here](#Configure-Local-Environment

macOS or Linux users: It is recommended to use the [Homebrew system](https://brew.sh/) to simplify the Miniconda installation process. Usage of Homebrew is explanained next. 

#### Do you have Homebrew Installed?

In your command prompt (i.e. *Terminal*) use a statement such as: `brew help`

If this errored, move on to the next section.

If this returned output (e.g. examples of usage) then you have Homebrew installed and can proceed to install conda [found here](#Install-Miniconda-with-Homebrew).

#### Install Homebrew

In your command prompt, call: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

#### Install Miniconda with Homebrew

In your command prompt, call: `brew install --cask miniconda`

When in doubt, calling in the `brew doctor` might help :pill: 

#### A Few Possibly Useful Conda Commands

All environment related commands can be found here: [Anaconda Environment Commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Here are a few of the most used ones though: 

List all environments (current environment as marked by the \*): `conda env list`

Create a new environment: `conda create --name myenv`

Activate an environment: `conda activate myenv`

Deactivate an environment and go back to system base: `conda deactivate`

List all installed packages for current environment: `conda list`

### Configure Local Environment

Using the command prompt, navigate to the local project repository directory -- On macOS, I recommend typing `cd ` in Terminal and then dragging the project folder from finder into Terminal. 

In your command prompt, call: `conda env create -f environment.yml`. This will create a new Conda virtual environment with the name: `regularized-regression-from-scratch`.

Activate the new environment by using: `regularized-regression-from-scratch`

### Access Project

After having activated your environment, use `jupyter notebook` to launch a Jupyter session in your browser. 

Within the Jupyter Home page, navigate and click on `nb.ipynb` in the list of files. This will launch a local kernel running the project notebook in a new tab. 

---
<br></br>
<br></br>
<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

<a name="footnote1">1</a>: This project was created on macOS version 11.0.1 (Big Sur) using Conda version 4.9.2, and Python 3.8 (please reach out to me if you need further system specifications). 

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
