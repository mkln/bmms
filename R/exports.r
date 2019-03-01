library(magrittr)
library(plyr)

#' Bayesian Modular & Multiscale Scalar-on-image regression
#' 
#' This function applies a Bayesian Modular & Multiscale regression model to
#' a scalar response, using image predictors.
#' @param y Vector of outcomes
#' @param X Array of dimension (p1, p2, n)
#' @param K Number of modular steps
#' @param mcmc Number of Markov chain-Monte Carlo iterations
#' @param burn Number of MCMC iterations to discard
#' @param family Assumes the outcome is binary with values in (0,1) if set to anything other than "gaussian"
#' @export
soir <- function(y, X, K, mcmc, burn=0, family="gaussian"){
  p1 <- nrow(X)
  p2 <- ncol(X)
  splits <- index_to_subscript((1:K)-1, matrix(0, nrow=p1, ncol=p2)) %>% 
    plyr::alply(2, function(x) t(as.matrix(x)))
  
  mask_forbid <- matrix(1, nrow=p1, ncol=p2)
  radius <- floor(max(p1,p2)/5)
  
  if(family=="gaussian"){
    result <- soi_cpp(y-mean(y), X, splits, mask_forbid,
                      1, 1, mcmc, burn, radius, 
                      0, 0, F, F)
  } else {
    result <- soi_binary_cpp(y, X, splits, mask_forbid,
                      1, 1, mcmc, burn, 2, 
                      0, 0, F, F)
    
  }
  return(result)
}

#' Bayesian Modular & Multiscale Scalar-on-function regression
#' 
#' This function applies a Bayesian Modular & Multiscale regression model to
#' a scalar response, using functional predictors recorded on a common, regular grid.
#' @param y Vector of outcomes
#' @param X Matrix of dimension (n, p)
#' @param K Number of modular steps
#' @param mcmc Number of Markov chain-Monte Carlo iterations
#' @param burn Number of MCMC iterations to discard
#' @param family Assumes the outcome is binary with values in (0,1) if set to anything other than "gaussian"
#' @export
sofr <- function(y, X, K, mcmc, burn=0, lambda=1, family="gaussian"){
  p <- ncol(X)
  splits <- (1:(p-1)-1) %>% sample(K) %>% lapply(function(x) x)

  if(family=="gaussian"){
    #cat(lambda)   
    result <- sofk(y, X, splits, mcmc, burn, lambda)
  } else {
    result <- sofk_binary(y, X, splits, mcmc, burn, lambda)
  }
  return(result)
}

