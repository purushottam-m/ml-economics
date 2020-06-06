
##################################################################################################################
#  
#  Description: Functions for BLP, GATES and CLAN analysis
#       The following file contains the functions required to estimate the Best Linear Predictor (BLP),
#       Sorted Group Average Treatment Effects (GATES) and Classification Analysis (CLAN) based on the 
#       paper by Chernozhukov et. al. (2018). We refer to the code by Hannah Bull at 
#`      https://ml-in-econ.appspot.com/lab3.html and the code by Chernozhukov et. al. (2018) at
#       https://github.com/demirermert/MLInference`. 
#
#
#  Author: Joshi Mridul (M2, APE), Ucidami Mafeni (M2, PPD) and Purushottam Mohanty (M2, APE)
#  Date Modified: 05/06/2020
#
##################################################################################################################



gates <- function(Y, W, X, Q=4, prop_scores=F) {
  
  ### Split the dataset into two sets, 1 and 2 (50/50)
  split <- createFolds(1:length(Y), k=2)[[1]]
  
  Ya = Y[split]
  Yb = Y[-split]
  
  Xa = X[split]
  Xb = X[-split]
  
  Wa = W[split, ]
  Wb = W[-split, ]
  
  ### Propensity score: On set A, train a model to predict X using W. Predict on set B.
  if (prop_scores==T) {
    sl_w1 = SuperLearner(Y = Xa, 
                         X = Wa, 
                         newX = Wb, 
                         family = binomial(), 
                         SL.library = "SL.xgboost", 
                         cvControl = list(V=0))
    
    p <- sl_w1$SL.predict
  } else {
    p <- rep(mean(Xa), length(Xb))
  }
  
  ### let D = W(set B) - propensity score.
  D <- Xb-p
  
  ### Get CATE (for example using causal forests) on set A. Predict on set B.
  tree_fml <- as.formula(paste("Y", paste(names(Wa), collapse = ' + '), sep = " ~ "))
  cf <- causalForest(tree_fml,
                     data=data.frame(Y=Ya, Wa), 
                     treatment=Xa, 
                     split.Rule="CT", 
                     split.Honest=T,  
                     split.Bucket=T, 
                     bucketNum = 5,
                     bucketMax = 100, 
                     cv.option="CT", 
                     cv.Honest=T, 
                     minsize = 2, 
                     split.alpha = 0.5, 
                     cv.alpha = 0.5,
                     sample.size.total = floor(nrow(Wa) / 2), 
                     sample.size.train.frac = .5,
                     mtry = ceiling(ncol(Wa)/3), 
                     nodesize = 5, 
                     num.trees = 10, 
                     ncov_sample = ncol(Wa), 
                     ncolx = ncol(Wa))
  
  cate_cf <- predict(cf, newdata = Wb, type="vector")
  
  
  ############# Best Linear Predictor #############
  
  diff <- cate_cf - mean(cate_cf)
  B2 <-  diff*D
  
  ### STEP 4: Create a dataframe with Y, W (set B), D and G. Regress Y on group membership variables and covariates. 
  df <- data.frame(Y=Yb, Wb, D, B2, p)
  
  Wnames <- paste(colnames(Wb), collapse="+")
  fml <- paste("Y ~",Wnames,"+ D + B2")
  modelBLP <- lm(fml, df, weights = 1/(p*(1-p))) 
  
  
  ############# GATES ##########
  
  ### Divide the cate estimates into Q tiles, and call this object G. 
  G <- data.frame(cate_cf) %>% # replace cate_cf with the name of your predictions object
    ntile(Q) %>%  # Divide observations into Q-tiles
    factor()
  
  ### Create a dataframe with Y, W (set B), D and G. Regress Y on group membership variables and covariates. 
  df <- data.frame(Y=Yb, G, Wb, D, p)
  
  Wnames <- paste(colnames(Wb), collapse="+")
  fml <- paste("Y ~",Wnames,"+ D:G")
  modelGATES <- lm(fml, df, weights = 1/(p*(1-p))) 
  
  models <- list(modelBLP, modelGATES)
  names(models) <- c("BLP", "GATES")
  
  return(models) 
}


###########

BLP_GATES_CLAN <- function(models) {
  
  # BLP
  thetahat <- models[["BLP"]] %>% 
    .$coefficients %>%
    .[c("D","B2")]
  
  # Confidence intervals
  cihat <- confint(models[["BLP"]])[c("D","B2"),]
  
  resBLP <- tibble(depvar = c(depvar, depvar),
                coefficient = c("ATE","HET"),
                estimates = thetahat,
                ci_lower_90 = cihat[,1],
                ci_upper_90 = cihat[,2])
  
  
  # GATES
  thetahat <- models[["GATES"]] %>% 
    .$coefficients %>%
    .[c("D:G1","D:G2","D:G3","D:G4")]
  
  # Confidence intervals
  cihat <- confint(models[["GATES"]])[c("D:G1","D:G2","D:G3","D:G4"),]
  
  resGATES <- tibble(depvar = c(depvar, depvar, depvar, depvar),
                coefficient = c("gamma1","gamma2","gamma3","gamma4"),
                estimates = thetahat,
                ci_lower_90 = cihat[,1],
                ci_upper_90 = cihat[,2])
  
  
  # CLAN
  # regression dataset
  clandta <- models[["GATES"]]$model %>%
    filter(G==1 | G==4) %>%
    mutate(dummy_tb = ifelse(G==4,1,0))
  
  resCLAN <- NULL
  
  for (indvar in vec_indvar) {
    
    table <- data.frame(1:4)
    
    form <- as.formula(paste0(indvar, "~", "dummy_tb"))
    clan_reg <- lm(form, data = clandta)
    
    coef <- summary(clan_reg)$coefficients[2,1] 
    cfl  <- confint(clan_reg)[2,1]
    cfu  <- confint(clan_reg)[2,2]
    pval <- summary(clan_reg)$coefficients[2,4]*2
    
    coeftable <- data.frame(depvar, indvar, coef, cfl, cfu, pval)
    names(coeftable) <- c("depvar", "indvar", "coef", "cfl", "cfu", "pval")
    coeftable$depvar <- as.character(coeftable$depvar)
    coeftable$indvar <- as.character(coeftable$indvar)
    
    resCLAN <- rbind(resCLAN, coeftable)
    
  }
  
  finaltable <- list(resBLP, resGATES, resCLAN)
  names(finaltable) <- c("BLP", "GATES", "CLAN")
  
  return(finaltable)
}



