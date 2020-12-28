# Machine Learning for Economics

## Double Debiased Machine Learning
File k2ml.R provides a generalized function to implement the double debiased machine learning using K-sample splits. It is based on the paper - *Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters*. The function is written with reference to the code by Hannah Bull (Paris School of Economics (PSE)) available at (https://ml-in-econ.appspot.com/lab2.html). 

To use the k2ml.R function the following packages needs to be installed:
```
if (!require(pacman)) install.packages('pacman', repos = 'https://cran.rstudio.com')
pacman:p_load(SuperLearner, tidyverse, clusterGeneration, mvtnorm, xgboost)
```

## Generic ML Inference on Heterogenous Treatment Effects in Randomized Experiments
File 1_hetML.R provides functions to estimate the Best Linear Predictor (BLP), Sorted Group Average Treatment Effects (GATES) and Classification Analysis (CLAN) based on the paper by Chernozhukov et. al. (2018). The code has been written by Joshi Mridul (PSE), Ucidami Mafeni (PSE), and Purushottam Mohanty (PSE). We refer to the code by Hannah Bull at (https://ml-in-econ.appspot.com/lab3.html) and the code by Chernozhukov et. al. (2018) at (https://github.com/demirermert/MLInference). 

File 2_hetML.R is an implementation of the function where we replicate Table 6 of the paper by Hirshleifer, et. al (2016) using machine learning methods. Correspondingly, we estimates the Best Linear Predictor (BLP), Sorted Group Average Treatment Effects (GATES) and Classification Analysis (CLAN). The file 2_hetML.R runs the function BLP_GATES_CLAN() and provides well formated final tables for BLP, GATES and CLAN and exports them in Latex. It also exports plots for the GATES coefficients. 

The following packages needs to installed in order to run the code:
```
if (!require(pacman)) install.packages('pacman', repos = 'https://cran.rstudio.com')
pacman::p_load(tidyverse, stats, readstata13, foreach, data.table, lmtest, sandwich, stargazer, xtable,
               glmnet, randomForest, devtools, knitr, SuperLearner, caret, xgboost, gridExtra)
```
