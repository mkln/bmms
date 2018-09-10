//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef bayes_lm
#define bayes_lm

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <random>
#include <ctime>
#include <math.h>
#include <cstdlib>
#include <Rcpp.h>


//double rndpp_gamma(double alpha, double beta);
//arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma);

class BayesLM{
public:
  // data
  arma::vec y;
  arma::vec ycenter;
  arma::mat X;
  int n;
  int p;
  bool fix_sigma;
  
  // useful
  arma::mat XtX;
  arma::mat XtXi;
  double yty;
  
  // model: y = Xb + e
  // where e is Normal(0, sigmasq I_n)
  double icept;
  arma::vec b;
  double sigmasq;
  double lambda; // ridge
  
  // priors
  
  // inverse gamma for sigma
  double alpha;
  double beta;
  
  // mean and variance for Normal for b
  arma::vec m;
  arma::mat M;
  arma::mat Mi;
  double mtMim;
  
  // posterior
  
  double alpha_n;
  double beta_n;
  arma::vec mu;
  arma::mat Sigma;
  double mutSimu;
  
  arma::mat inv_var_post;
  
  arma::vec reg_mean;
  arma::vec reg_mean_prior;
  
  void posterior();
  void beta_sample();
  void lambda_update(double);
  void chg_y(arma::vec&);
  void chg_data(arma::vec&, arma::mat&);
  
  BayesLM();
  BayesLM(const arma::vec&, const arma::mat&, bool);
  BayesLM(arma::vec, arma::mat, double);
  BayesLM(const arma::vec&, const arma::mat&, double, bool);
  BayesLM(arma::vec, arma::mat, arma::mat);
};


class BayesLMg{
public:
  // data
  arma::vec y;
  arma::vec ycenter;
  arma::mat X;
  int n;
  int p;
  bool sampling_mcmc, fix_sigma;
  
  // useful
  arma::mat XtX;
  arma::mat Mi; // inverse of prior cov matrix. b ~ N(0, sigmasq * Mi^-1)
  double yty;
  arma::mat In;
  
  // model: y = Xb + e
  // where e is Normal(0, sigmasq I_n)
  double icept;
  arma::vec b;
  double sigmasq;
  double g; // ridge
  
  // priors
  // inverse gamma for sigma
  double alpha;
  double beta;
  
  // posterior
  
  double alpha_n;
  double beta_n;
  arma::vec mu;
  arma::mat Sigma;
  double mutSimu;
  
  arma::mat inv_var_post;
  
  arma::vec reg_mean;
  
  double yPxy;
  double marglik;
  double get_marglik(bool);
  
  void sample_sigmasq();
  void sample_beta();
  
  void change_X(const arma::mat&);
  
  BayesLMg();
  //BayesLMg(const arma::vec&, const arma::mat&, bool);
  //BayesLMg(arma::vec, arma::mat, double);
  BayesLMg(const arma::vec&, const arma::mat& , double, bool, bool);
  //BayesLMg(arma::vec, arma::mat, arma::mat);
};

class BayesSelect{
public:
  // data
  bool fix_sigma;
  int p, n;
  double icept, g, yty, yPxy, alpha, beta, marglik;
  arma::vec y, ycenter;
  
  void change_X(const arma::mat&);
  double get_marglik(bool);
  
  BayesSelect();
  //BayesLMg(const arma::vec&, const arma::mat&, bool);
  //BayesLMg(arma::vec, arma::mat, double);
  BayesSelect(const arma::vec&, const arma::mat&, double, bool);
  //BayesLMg(arma::vec, arma::mat, arma::mat);
};


class VarSelMCMC{
public:
  arma::vec y;
  arma::mat X;
  int n, p;
  
  arma::uvec gamma;
  arma::uvec gamma_proposal;
  BayesSelect model;
  BayesSelect model_proposal;
  BayesLMg sampled_model;
  
  //arma::vec gamma_start_prior; //prior prob of values to start from
  
  arma::vec sampling_order;
  
  int mcmc;
  
  void chain();
  
  arma::mat gamma_stored;
  arma::mat beta_stored;
  arma::vec sigmasq_stored;
  
  VarSelMCMC(const arma::vec&, const arma::mat&, const arma::vec&, double, double, bool, int);
};


Rcpp::List my_bayeslm(arma::vec y, arma::mat X);

#endif