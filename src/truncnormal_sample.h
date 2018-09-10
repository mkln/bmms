//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef truncnormal
#define truncnormal

#include <RcppArmadillo.h>

using namespace std;

arma::vec pnorm01_vec(const arma::vec& x, int lower, int logged);


arma::vec qnorm01_vec(const arma::vec& x, int lower, int logged);

arma::uvec usetdiff(const arma::uvec& x, const arma::uvec& y);

arma::vec lnNpr_cpp(const arma::vec& a, const arma::vec& b);

arma::field<arma::mat> cholperm_cpp(arma::mat Sig, arma::vec l, arma::vec u);

arma::mat gradpsi_cpp(const arma::vec& y, const arma::mat& L, 
                      const arma::vec& l, const arma::vec& u, arma::vec& grad);

arma::vec armasolve(arma::mat A, arma::vec grad);

arma::vec nleq(const arma::vec& l, const arma::vec& u, const arma::mat& L);


arma::vec ntail_cpp(const arma::vec& l, const arma::vec& u);


arma::vec trnd_cpp(const arma::vec& l, const arma::vec& u);

arma::vec tn_cpp(const arma::vec& l, const arma::vec& u);

arma::vec trandn_cpp(const arma::vec& l, const arma::vec& u);

arma::mat mvnrnd_cpp(int n, const arma::mat& L, 
                     const arma::vec& l, const arma::vec& u, 
                     arma::vec mu, arma::vec& logpr);

double psy_cpp(arma::vec x, const arma::mat& L, 
               arma::vec l, arma::vec u, arma::vec mu);

arma::mat mvrandn_cpp(const arma::vec& l_in, const arma::vec& u_in, 
                      const arma::mat& Sig, int n);

arma::mat rndpp_mvnormal2(int n, const arma::vec& mu, const arma::mat& sigma);

arma::mat rndpp_mvnormalnew(int n, const arma::vec &mean, const arma::mat &sigma);
arma::mat mvtruncnormal(const arma::vec& mean, 
                        const arma::vec& l_in, const arma::vec& u_in, 
                        const arma::mat& Sig, int n);
arma::mat mvtruncnormal_eye1(const arma::vec& mean, 
                             const arma::vec& l_in, const arma::vec& u_in);
arma::mat get_S(const arma::vec& y, const arma::mat& X);

arma::mat get_Ddiag(const arma::mat& Sigma, const arma::mat& S);

arma::mat diag_default_mult(const arma::mat& A, const arma::vec& D);
arma::mat diag_custom_mult(const arma::mat& A, const arma::vec& D);
arma::mat beta_post_sample(const arma::vec& mu, const arma::mat& Sigma,
                                const arma::mat& S, const arma::vec& Ddiag, int sample_size);

#endif