library(tidyverse)
library(magrittr)
library(mvtnorm)
library(glue)
library(grid)
library(gridExtra)
library(latex2exp)
library(waveband)
library(bmms)

p <- 512

prepare_saved_splits <- function(p, l){
  starting_splits <- list(
    c(p/3, 2*p/3),
    c(p/4, 3*p/4),
    c(p/8, 7*p/8),
    c(3*p/8, 5*p/8),
    c(1, p-2),
    c(p/16, 15*p/16)
  ) %>% lapply(function(x) (x-1) %>% round())
  starting_splits <- starting_splits[1:l]
  save(list=c("starting_splits"), file=glue("sim/starting_splits_{p}.RData"))
  return(starting_splits)
}

rho <- .2
sigmasq <- 1
nr <- 200
nout <- 100
gg <- nr
s <- 1

gg <- nr
# PARAMS

mr <- 1
expcovmat <- outer(seq(1, p), seq(1,p), function(x,y)  mr*exp(- (1-rho) * abs(x-y))) 
# GENERATE NEW DATA
Xp <- rmvnorm(nr, rep(0, p), expcovmat)
Xout <- rmvnorm(nout, rep(0, p), expcovmat)

beta <- waveband::test.data(type="blocks", n=p)$y 

y <- Xp %*% beta + rnorm(nr, 0, sigmasq^.5)
yout <- Xout %*% beta + rnorm(nout, 0, sigmasq^.5)

# lambda_in: penalty on number of jumps of the step functions (higher-resolutions get stronger penalty)
# ssr: penalty on proximity of jump locations
cvpar <- list(lambda_in = 1, ssr=1)

# MODULAR MCMC
mcmc <- 300
keep <- min(4000, mcmc*.8)
(burn <- max(mcmc-keep, 0))

K <- 3
starting_splits <- prepare_saved_splits(p, K)
# y: (nr) numeric
# Xp: (nr, p) matrix
# starting_splits: a list made of K vectors. each vector with any number of entries between 0 and p-1. MCMC will change everything except K.
# mcmc: total iterations
# burn: not saved
blocks_out <- sofk(y, Xp, starting_splits, 
                   mcmc, burn, lambda=cvpar$lambda_in, 
                   .01, .01, # prior params for inverse gamma for sigmasq 
                   0, 0, # utility params, leave at 0
                   F, # use a single sigmasq for all scales?
                   T, # silent = !verbose
                   gg, # g-prior on coefficients
                   cvpar$ssr, 
                   T # smoothing parameter (called "radius") is sampled from mcmc. if F, this is fixed at 0
                   )

## final estimate:
# blocks_out$theta %>% apply(2, mean) %>% plot(type="l")
## lowest resolution:
# blocks_out$theta_ms %>% sapply(function(x) x[[1]]) %>% apply(1, mean) %>% plot(type="l")
## contribution of second resolution:
# blocks_out$theta_ms %>% sapply(function(x) x[[2]] - x[[1]]) %>% apply(1, mean) %>% plot(type="l")
# see plotting code below for additional output

rho <- .2
expcovmat <- outer(seq(1, p), seq(1,p), function(x,y)  mr*exp(- (1-rho) * abs(x-y))) 
Xp <- rmvnorm(nr, rep(0, p), expcovmat)
beta <- waveband::test.data(type="doppler", n=p)$y 

y <- Xp %*% beta + rnorm(nr, 0, sigmasq^.5)
yout <- Xout %*% beta + rnorm(nout, 0, sigmasq^.5)

doppler_out <- sofk(y, Xp, starting_splits, 
                    mcmc, burn, lambda=cvpar$lambda_in, 
                    .01, .01, 0, 0, F, T, gg, cvpar$ssr, T)

resplot <- function(tdf, sdf){
  thetaplot1 <- ggplot(tdf, aes(x=time)) +
    geom_ribbon(aes(ymin=lower, ymax=upper), fill="#111E6C", alpha=0.3) +
    geom_line(aes(y=center)) +
    facet_grid(name ~ ., labeller=label_parsed) +
    theme_bw() + 
    theme(axis.text.y = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          plot.margin = unit(c(2,2,2,2), "pt")) +
    ylab(NULL) + xlab(NULL)
  
  densplot1 <- ggplot(sdf, aes(center)) + 
    facet_grid(name ~., labeller=label_parsed) +
    geom_density(fill="#FC6600", alpha=.5) +
    theme_bw() + 
    theme(axis.text.y = element_blank(),
          plot.margin = unit(c(2,2,4,2), "pt"),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank()) +
    ylab(NULL) + xlab(NULL) + xlim(0, 1)
  
  res1plot <- grid.arrange(thetaplot1, densplot1, nrow=2, heights=c(2,1))
  return(res1plot)
}

plotter <- function(bmms_out, title, beta){
  splits1 <- bmms_out$splits %>% lapply(function(x) x[[1]]) %>% unlist()
  splits2 <- bmms_out$splits %>% lapply(function(x) x[[2]]) %>% unlist()
  splits3 <- bmms_out$splits %>% lapply(function(x) x[[3]]) %>% unlist()
  
  # beta
  bmms_est <- bmms_out$theta %>% apply(2, mean)
  bmms_est_low <- bmms_out$theta %>% apply(2, function(x) quantile(x, 0.005))
  bmms_est_high <- bmms_out$theta %>% apply(2, function(x) quantile(x, 0.995))
  betadf <- data.frame(
    time = (1:p)/p,
    center = bmms_est,
    lower = bmms_est_low,
    upper = bmms_est_high,
    orig = beta,
    name = "beta[3](t)"
  )
  
  betaplot <- ggplot(betadf, aes(x=time)) +
    geom_ribbon(aes(ymin=lower, ymax=upper), fill="#111E6C", alpha=0.3) +
    geom_line(aes(y=center)) +
    geom_line(aes(y=orig), linetype="dashed", color="black", alpha=0.3) +
    theme_bw() + 
    theme(axis.text.y = element_blank(),
          plot.margin = unit(c(5,2,4,2), "pt")) +
    ylab(NULL) + xlab(NULL) +
    ggtitle(TeX(glue("BM&Ms for $\\beta(t)$ : {title}")))
  
  bmms_intercept <- mean(y - Xp %*% bmms_est)
  mean((bmms_est-beta)^2)
  mean((yout - Xout %*% bmms_est - bmms_intercept)^2)
  
  it <- 5
  mt <- 2
  # 1
  theta1 <- bmms_out$theta_ms %>% sapply(function(x) x[[1]]) %>% apply(1, mean)
  theta1_low <- bmms_out$theta_ms %>% sapply(function(x) x[[1]]) %>% apply(1, function(x) quantile(x, 0.005))
  theta1_high <- bmms_out$theta_ms %>% sapply(function(x) x[[1]]) %>% apply(1, function(x) quantile(x, 0.995))
  t1df <- data.frame(
    time = (1:p)/p,
    center = theta1,
    lower = theta1_low,
    upper = theta1_high,
    name = "theta[1](t)"
  )
  t1df[,c("center", "lower", "upper")] <- it + mt * t1df[,c("center", "lower", "upper")] 
  
  # 2
  theta2_all <- bmms_out$theta_ms %>% sapply(function(x) x[[2]]) - bmms_out$theta_ms %>% sapply(function(x) x[[1]]) 
  theta2 <- theta2_all %>% apply(1, mean)
  theta2_low <- theta2_all %>% apply(1, function(x) quantile(x, 0.005))
  theta2_high <- theta2_all %>% apply(1, function(x) quantile(x, 0.995))
  t2df <- data.frame(
    time = (1:p)/p,
    center = theta2,
    lower = theta2_low,
    upper = theta2_high,
    name = "theta[2](t)"
  )
  t2df[,c("center", "lower", "upper")] <- it + mt * t2df[,c("center", "lower", "upper")] 
  
  # 3
  theta3_all <- bmms_out$theta_ms %>% sapply(function(x) x[[3]]) - bmms_out$theta_ms %>% sapply(function(x) x[[2]]) 
  theta3 <- theta3_all %>% apply(1, mean)
  theta3_low <- theta3_all %>% apply(1, function(x) quantile(x, 0.005))
  theta3_high <- theta3_all %>% apply(1, function(x) quantile(x, 0.995))
  t3df <- data.frame(
    time = (1:p)/p,
    center = theta3,
    lower = theta3_low,
    upper = theta3_high,
    name = "theta[3](t)"
  )
  t3df[,c("center", "lower", "upper")] <- it + mt * t3df[,c("center", "lower", "upper")] 
  
  # 1
  s1df <- data.frame(
    center = c(splits1)/p,
    name = "Pr(t %in% S[2])")
  
  # 2
  splits2 <- bmms_out$splits %>% lapply(function(x) x[[2]]) %>% unlist()
  s2df <- data.frame(
    center = c(splits1,splits2)/p,
    name = "Pr(t %in% S[2])")
  
  # 3
  splits3 <- bmms_out$splits %>% lapply(function(x) x[[3]]) %>% unlist()
  s3df <- data.frame(
    center = c(splits1,splits2,splits3)/p,
    name = "Pr(t %in% S[3])")
  
  res1plot <- resplot(t1df, s1df)
  res2plot <- resplot(t2df, s2df)
  res3plot <- resplot(t3df, s3df)
  
  theta_plot <- grid.arrange(res1plot, res2plot, res3plot, ncol=1)
  
  return(list(theta_plot, betaplot))
}

blocks_plots <- blocks_out %>% plotter("Blocks", waveband::test.data(type="blocks", n=p)$y )
blocks_out$radius %>% mean

doppler_plots <- doppler_out %>% plotter("Doppler", waveband::test.data(type="doppler", n=p)$y )
doppler_out$radius %>% mean

testplot <- grid.arrange(grid.arrange(blocks_plots[[1]], top=textGrob(TeX("Blocks: $\\theta_j$ and $S_j$"), gp=gpar(fontsize=15,font=1))), 
                         grid.arrange(blocks_plots[[2]], doppler_plots[[2]], ncol=1),
                         grid.arrange(doppler_plots[[1]], top=textGrob(TeX("Doppler: $\\theta_j$ and $S_j$"), gp=gpar(fontsize=15,font=1))), 
                         widths = c(1,1.5,1))