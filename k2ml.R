
# Double Debiased Machine Learning with K-sample splitting

k2ml <- function(X, W, Y, K, SL.library.X = "SL.lm",  SL.library.Y = "SL.lm", 
                     family.X = gaussian(), family.Y = gaussian()) {

  ### Split data into K parts
  set.seed(123)
  split <- split(sample(seq_len(length(Y))), 
                 rep(1:K, length.out = length(Y), each = ceiling(length(Y)/K)))
  
  # single list containing K sample splits of my_data
  my_data_list <- lapply(c(1:K), function(i) 
    list(y = Y[split[[i]]], x = X[split[[i]]], w = W[split[[i]],]))

  # Use SuperLearner to train model for E[X|W] using all samples except kth sample and then predict X on kth
  # sample using this model. Do the same for all the k samples.
  sl_x <- lapply(c(1:K), function(i) {
    SuperLearner(Y = X[-split[[i]]], 
                 X = data.frame(w = W[-split[[i]],]), # the data used to train the model
                 newX = data.frame(w = W[split[[i]],]), # the data used to predict x
                 family = family.X, 
                 SL.library = SL.library.X,
                 cvControl = list(V=0))
      }
    )
  
  # Predictions for X on all sets
  x_hat <- lapply(c(1:K), function(i) sl_x[[i]]$SL.predict)
  
  # Residuals X - X_hat on all sets
  res_x <- lapply(c(1:K), function(i) my_data_list[[i]]$x - x_hat[[i]])
  
  
  # Use SuperLearner to train model for E[Y|W] using all samples except kth sample and then predict Y on kth
  # sample using this model. Do the same for all the k samples.
  sl_y <- lapply(c(1:K), function(i) {
    SuperLearner(Y = Y[-split[[i]]], 
                       X = data.frame(w = W[-split[[i]],]), # the data used to train the model
                       newX= data.frame(w = W[split[[i]],]), # the data used to predict x
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0)) 
      }
    )
  
  # Predictions for Y on all sets
  y_hat <- lapply(c(1:K), function(i) sl_y[[i]]$SL.predict)
  
  # Residuals Y - Y_hat on all sets
  res_y <- lapply(c(1:K), function(i) my_data_list[[i]]$y - y_hat[[i]])
  
  # Regress (Y - Y_hat) on (X - X_hat) for each set and get the k coefficients of (X - X_hat)
  beta_list <- lapply(c(1:K), function(i) (mean(res_x[[i]]*res_y[[i]])/(mean(res_x[[i]]**2))))
  
  # Take the average of these k coefficients from the k sets (= beta)
  beta <- mean(unlist(beta_list))
  
  # Compute standard errors. This is just the usual OLS standard errors in the regression res_y = res_x*beta + eps. 
  psi_stack = unlist(lapply(c(1:K), function(i) res_y[[i]] - res_x[[i]]*beta))
  res_stack = unlist(res_x)
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(Y))
  
  return(c(beta,se))
}
