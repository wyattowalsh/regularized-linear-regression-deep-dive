<h1 align="center"> Regularized Linear Regression Deep Dive: <br> Application to Wine Quality Regression Dataset </h1>

[![View on Medium](https://img.shields.io/badge/Medium-View%20on%20Medium-red?logo=medium)](https://towardsdatascience/tagged/regularized-regression)

This project consists of a deep dive on multiple linear regression (OLS) and its regularized variants (Ridge, the Lasso, and the Elastic Net) as well as Python implementations for exploratory data analysis, K-Fold cross-validation and modeling functions as applied to regression of a wine quality dataset. This examination applies optimization theory to either derive model estimator (for OLS and Ridge) or derive the update rule for Pathwise Coordinate Descent (the discrete optimization algorithm chosen and implemented to solve the Lasso and the Elastic Net). These derivations have accompanying Python implementations, which are leveraged to predict wine quality ratings within a supervised learning context.

<p align="center">
  A three-part series of blog posts was published in <a href='https://towardsdatascience.com/'><b><i>Towards Data Science</i></b></a> <br> Read them here:
  <a href='https://towardsdatascience/tagged/regularized-regression'><img src='https://img.shields.io/badge/Medium-View%20on%20Medium-red?logo=medium'/>
<!--   <a href='https://towardsdatascience.com/regularized-linear-regression-models-57bbdce90a8c'><b>Part One</b></a>, <a href='https://towardsdatascience.com/regularized-linear-regression-models-44572e79a1b5'><b>Part Two</b></a>, and <a href='https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9'><b>Part Three</b></a> -->
 <br><br>
</p>

<p align="center">
  Interact with the project notebook in your web browser using the <i>Binder</i> service  
<a href=https://mybinder.org/v2/gh/wyattowalsh/regularized-linear-regression-deep-dive/HEAD?filepath=nb.ipynb> <img src=https://mybinder.org/badge_logo.svg></a>
 <br><br>
</p>

<p align="center">
    <a href=#explanation-of-repository-contents>Explanation of Repository Contents</a>  ·
    <a href=#technical-overview> Technical Overview</a>  ·
    <a href=#installation-instructions> Installation Instructions</a> 
  <br><br>
</p>

![](notebook_gif.gif)
---

## Explanation of Repository Contents 

- `data/` - contains the project's wine quality dataset 
- `src/` - holds all the project source code
- `nb.ipynb`  - project notebook
- `environment.yml` - Conda virtual environment reproduction file

---

## Technical Overview  

The entirety of this project is written in Python (version 3.8) with a majority of functions depending on NumPy and several on pandas. Matplotlib and Seaborn are used for visualization. Furthermore, there are a few other simple dependencies used like the time or math libraries. 

Implementations can be found for train-test data splitting, variance inflation factor calculation, K-Fold cross-validation, ordinary least squares (OLS), Ridge, the Lasso, and the Elastic Net as well as several other functions used to produce the notebook. 

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

See [**here**](https://github.com/wyattowalsh/regularized-linear-regression-deep-dive/blob/master/SOURCES.md) for the different sources utilized to synthesize this project. 
