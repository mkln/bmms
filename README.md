# Bayesian Modular &amp; Multiscale Regression
M. Peruzzi and D. B. Dunson
https://arxiv.org/abs/1809.05935

A work-in-progress package for multiscale regression. 
Functions currently implemented for:
 * scalar-on-function (multiscale) regression and classification
 * scalar-on-image (multiscale) regression and classification
 * multiscale regression on nested predictors with no spatiotemporal structure

Some under-the-hood functions are imported from `mkln/emotionalabyss`. 

Installation: first `devtools::install_github("mkln/emotionalabyss")` then `devtools::install_github("mkln/bmms")`. 


*Abstract:* We tackle the problem of multiscale regression for predictors that are spatially or temporally indexed, or with a pre-specified multiscale structure, with a Bayesian modular approach. The regression function at the finest scale is expressed as an additive expansion of coarse to fine step functions. Our Modular and Multiscale (M&M) methodology provides multiscale decomposition of high-dimensional data arising from very fine measurements. Unlike more complex methods for functional predictors, our approach provides easy interpretation of the results. Additionally, it provides a quantification of uncertainty on the data resolution, solving a common problem researchers encounter with simple models on down-sampled data. We show that our modular and multiscale posterior has an empirical Bayes interpretation, with a simple limiting distribution in large samples. An efficient sampling algorithm is developed for posterior computation, and the methods are illustrated through simulation studies and an application to brain image classification.


## Sample output

Example output for scalar-on-function regression: 
![BM&Ms scalar-on-function](https://i.imgur.com/wwQreVM.png)

Code used to create the above plot is in `bmms_sofr_example_plot.r`. 

Video of an MCMC chain: https://youtu.be/oUb8ZPRC_IM 

Sorry, but at this time a complete documentation is also a work in progress.