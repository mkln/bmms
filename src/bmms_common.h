//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::interfaces(r, cpp)]]

#ifndef bmms_common
#define bmms_common

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <random>
#include <ctime>
#include <math.h>
#include <cstdlib>

arma::vec msample(arma::vec avail, int i, bool b, arma::vec p);
arma::vec bmms_setdiff(arma::vec& x, arma::vec& y);
arma::vec dmvnrm_arma(arma::mat x,  
                      arma::rowvec mean,  
                      arma::mat sigma, 
                      bool logd = false);
arma::vec dmvnrm_s1diag(arma::mat x,  
                        arma::rowvec mean,  
                        bool logd);
arma::mat X2Dgrid(arma::vec x1, arma::vec x2);
int rndpp_sample1_comp(arma::vec x, int p, int current_split, double decay);
int rndpp_sample1_comp_old(arma::vec x, int p);
int rndpp_unif_int(int max);
int rndpp_discrete(arma::vec probs);
double rndpp_gamma(double alpha, double beta);
double rndpp_normal(double mean, double sigma);
double rndpp_bern(double p);
arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma);
arma::vec nonzeromean(arma::mat mat_mcmc);
arma::vec col_eq_check(arma::mat A);
arma::vec col_sums(const arma::mat& matty);
arma::mat drop_dup_cols(arma::mat A);
arma::mat exclude(arma::mat test, arma::vec excl);
arma::mat single_split(arma::mat Jcoarse, int where, int p);
arma::mat multi_split(arma::mat Jcoarse, arma::vec where, int p);
arma::vec split_fix(arma::field<arma::vec>& in_splits, int stage);
arma::field<arma::vec> stage_fix(arma::field<arma::vec>& in_splits);
arma::field<arma::vec> split_shifter(arma::field<arma::vec> split_seq, int shift_size, int p);
double split_struct_ratio(arma::vec prop_split, arma::vec orig_split, int p, double param=3);
#endif
