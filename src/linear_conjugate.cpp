#include "linear_conjugate.h"
#include "bmms_common.h"
#include "metrop_helper.h"

using namespace std;
using namespace Rcpp;

//std::random_device rd;
//std::mt19937 mt(rd());

bool do_I_accept(double logaccept){ //, string name_accept, string name_count, List mcmc_pars){
  double acceptj = 1.0;
  if(!arma::is_finite(logaccept)){
    acceptj = 0.0;
  } else {
    if(logaccept < 0){
      acceptj = exp(logaccept);
    }
  }
  double u = runif(1)(0);
  if(u < acceptj){
    return true;
  } else {
    return false;
  }
}

const double EPS = 0.01;
const double tau_accept = 0.234;
const double g_exp = 0.7;
const int g0 = 200;
const double rho_max = 15;

void adapt_alg6_s(double param, arma::vec& sumparam, arma::vec& prodparam,
                  arma::vec& paramsd, arma::mat& sd_param, int mc, double accept_ratio){
  
  int j = 0;
  //int siz = param.n_rows;
  
  sd_param(mc+1, j) = sd_param(mc, j) + pow(mc+0.0, -g_exp) * (accept_ratio - tau_accept);
  if(sd_param(mc+1, j) > exp(rho_max)){
    sd_param(mc+1, j) = exp(rho_max);
  } else {
    if(sd_param(mc+1, j) < exp(-rho_max)){
      sd_param(mc+1, j) = exp(-rho_max);
    }
  }
  
  sumparam(j) = sumparam(j) + param; // param(j);
  prodparam(j) = prodparam(j) + param*param; //param(j) * param(j);
  
  if(mc > g0){
    paramsd(j) = sd_param(mc+1, j) / (mc-1.0) * (
      prodparam(j) - (sumparam(j)*sumparam(j))/(mc+0.0) ) +
        sd_param(mc+1, j) * EPS;
  }
}  


// [[Rcpp::export]]
arma::mat rndpp_stdmvnormal(int n, int dimension){
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  for(int i=0; i<n; i++){
    xtemp = Rcpp::rnorm(dimension, 0.0, 1.0);
    outmat.row(i) = xtemp.t();
  }
  return outmat;
}

// [[Rcpp::export]]
arma::mat rndpp_mvt(int n, const arma::vec &mu, const arma::mat &sigma, double df){
  double w=1.0;
  arma::mat Z = rndpp_stdmvnormal(n, mu.n_elem);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  arma::mat AZ = Z;
  for(int i=0; i<AZ.n_rows; i++){
    w = sqrt( df / R::rchisq(df) );
    AZ.row(i) = (mu + w * (cholsigma * Z.row(i).t())).t();
  }
  return AZ;
}


BayesLM::BayesLM(){
  
}


BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, bool fixs=false){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  lambda = 0.0;
  XtX = X.t() * X;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  m = arma::zeros(p);
  //M = n*XtXi;
  Mi = 1.0/n * XtX + arma::eye(p,p) * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 0.0;
  beta = 0.0;
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  sigmasq = 1.0;
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  
  reg_mean = icept + X * b;
}


BayesLM::BayesLM(arma::vec yy, arma::mat XX, double lambda_in = 1){
  fix_sigma = false;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  //clog << arma::size(XX) << endl;
  lambda = lambda_in;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  //clog << lambda << endl;
  
  XtX = X.t() * X;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  m = arma::zeros(p);
  //M = n*XtXi;
  Mi = 1.0/n * XtX + arma::eye(p,p) * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.1; // parametrization: a = mean^2 / variance + 2
  beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  alpha_n = alpha + n/2.0;
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
  
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  reg_mean = icept + X * b;
}


BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, double lambda_in = 1, bool fixs=false){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  //clog << arma::size(XX) << endl;
  lambda = lambda_in;
  XtX = X.t() * X;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  m = arma::zeros(p);
  //M = n*XtXi;
  Mi = 1.0/n * XtX + arma::eye(p,p) * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.1; // parametrization: a = mean^2 / variance + 2
  beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma) { 
    yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = 0.0;
    beta_n = 0.0;
    sigmasq = 1.0;
  }
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  reg_mean = icept + X * b;
}



BayesLM::BayesLM(arma::vec yy, arma::mat XX, arma::mat MM){
  // specifying prior for regression coefficient variance
  fix_sigma = false;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  lambda = 1.0;
  
  XtX = X.t() * X;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  m = arma::zeros(p);
  M = MM;
  Mi = arma::inv_sympd(M);
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.25; // parametrization: a = mean^2 / variance + 2
  beta = 0.625; //alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma) { 
    yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
    b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
    
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = 0.0;
    beta_n = 0.0;
    sigmasq = 1.0;
    b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  }
  
  reg_mean = icept + X * b;
}

void BayesLM::lambda_update(double lambda_new){
  // specifying prior for regression coefficient variance
  
  lambda = lambda_new;
  M = lambda_new*M/lambda;
  Mi = arma::inv_sympd(M);
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.25; // parametrization: a = mean^2 / variance + 2
  beta = 0.625; //alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  alpha_n = alpha + n/2.0;
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
  
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda_new, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  reg_mean = X * b;
}

void BayesLM::posterior(){
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
}

void BayesLM::beta_sample(){
  b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
}

void BayesLM::chg_y(arma::vec& yy){
  y = yy;
  icept = arma::mean(y);
  ycenter = y - icept;
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
  
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  reg_mean = icept + X * b;
}

void BayesLM::chg_data(arma::vec& yy, arma::mat& XX){
  // specifying prior for regression coefficient variance
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  XtX = X.t() * X;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  alpha_n = alpha + n/2.0;
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
  
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  reg_mean = icept + X * b;
}

//[[Rcpp::export]]
Rcpp::List my_bayeslm(arma::vec y, arma::mat X, bool fixs){
  
  BayesLM lmodel(y, X, 1.0, fixs);
  
  return Rcpp::List::create(
    Rcpp::Named("sigmasq") = lmodel.sigmasq,
    Rcpp::Named("beta") = lmodel.b
  );
}

// [[Rcpp::export]]
double lambda_prior_mhr(double lambda_now, double lambda_prop){
  double a=2.1;
  double b=1.1;
  return (a-1)*(log(lambda_prop) - log(lambda_now)) - (lambda_prop - lambda_now)/b;
}

//[[Rcpp::export]]
double lambda_lik_mhr(double lambda, double lambda_prop, int p, double mfact){
  // in log of course
  return p/2.0 * (log(lambda_prop) - log(lambda)) - (lambda_prop - lambda) * mfact;
}

//[[Rcpp::export]]
double lreg_lik_mhr(arma::vec &y, arma::mat &x, arma::vec b_prop, arma::vec b,
                    double sigmasq_prop, double sigmasq){
  return -y.n_elem/2.0 * (log(sigmasq_prop) - log(sigmasq)) - 
    1.0/(2*sigmasq_prop)*arma::conv_to<double>::from((y-x*b_prop).t()*(y-x*b_prop)) +
    1.0/(2*sigmasq)*arma::conv_to<double>::from((y-x*b).t()*(y-x*b));
}

//[[Rcpp::export]]
Rcpp::List bayes_ridge(arma::vec y, arma::mat X, int mcmc=1000, int burn=0){
  arma::vec beta = arma::zeros(X.n_cols);
  
  arma::mat priorM = arma::eye(X.n_cols, X.n_cols);
  
  double lambda=1;
  double lambda_prop=1;
  
  BayesLM lmodel(y, X, lambda*priorM);
  BayesLM propmodel = lmodel;
  
  double param_mult_temp=1;
  double prior_accept = 0;
  double logaccept=0;
  double mfact = 0;
  
  arma::vec lambda_save = arma::zeros(mcmc-burn);
  arma::vec sigmasq_save = arma::zeros(mcmc-burn);
  arma::mat beta_save = arma::zeros(X.n_cols, mcmc-burn);
  //arma::cube S_save = arma::zeros(X.n_cols, X.n_cols, mcmc-burn);
  
  // standard deviation of the proposals
  // will be adaptively calibrated
  arma::vec lambdasd = arma::ones(1);
  arma::vec sumlambda = arma::zeros(1);
  arma::vec prodlambda = arma::zeros(arma::size(lambdasd));
  arma::mat sd_lambda = arma::zeros(mcmc+1, 1);
  sd_lambda.row(0) += pow(2.4, 2);
  
  double accepted=0;
  double proposed=0;
  
  for(int m=0; m<mcmc; m++){
    //cout << "m " << m << endl;
    beta = lmodel.b;
    mfact = arma::conv_to<double>::from(1.0/(lmodel.sigmasq*2)*beta.t()*beta);
    
    // propose new lambda 
    param_mult_temp = Rcpp::rnorm(1)(0);
    param_mult_temp = exp(param_mult_temp * pow(lambdasd(0), 0.5));
    param_mult_temp = min(param_mult_temp, 1e10);
    param_mult_temp = max(param_mult_temp, 1e-10);
    
    lambda_prop = lambda * param_mult_temp;
    proposed++;
    
    propmodel.lambda_update(lambda_prop);
    
    prior_accept = lambda_prior_mhr(lambda, lambda_prop);
    
    logaccept = //lreg_lik_mhr(y, X, propmodel.b, lmodel.b, propmodel.sigmasq, lmodel.sigmasq) + 
      lambda_lik_mhr(lambda, lambda_prop, lmodel.p, mfact) + prior_accept;
    
    if(exp(logaccept) > runif(1)(0)){
      lambda = lambda_prop;
      lmodel = propmodel;  
      accepted++;
    }
    
    adapt_alg6_s(log(lambda), sumlambda, prodlambda, lambdasd, sd_lambda, 
                 m, accepted/proposed);
    
    if(1 & (mcmc > 50)){
      if(!(m % 50)){
        clog << "m: " << m << " " << accepted/proposed << " " << lambdasd(0) << endl;
      } 
    }
    
    if(m > burn-1){
      int i = m-burn;
      lambda_save(i) = lambda;
      beta_save.col(i) = beta;
      sigmasq_save(i) = lmodel.sigmasq;
      //S_save.slice(i) = S;
    }
  }
  cout << "acceptance ratio " << accepted/proposed << endl;
  
  //return (rndpp_mvnormal(1, b_mean_post, Cov_post)).t();
  return(Rcpp::List::create(
      Rcpp::Named("beta") = beta_save,
      Rcpp::Named("lambda") = lambda_save,
      Rcpp::Named("sigmasq") = sigmasq_save
  ));
}

//[[Rcpp::export]]
arma::mat hat_alt(const arma::mat& X){
  arma::mat U;
  arma::vec s;
  arma::mat V;
  int c = X.n_cols > X.n_rows ? X.n_rows : X.n_cols;
  arma::svd(U,s,V,X);
  return U.cols(0,c-1) * U.cols(0,c-1).t();
}

//[[Rcpp::export]]
arma::mat hat(const arma::mat& X){
  if(X.n_cols > X.n_rows){
    return hat_alt(X);
  } else {
    arma::mat iv;
    bool gotit = arma::inv_sympd(iv, X.t() * X);
    if(gotit){
      return X * iv * X.t();
    } else { 
      return hat_alt(X);
    }
  }
}



// log density of mvnormal mean 0
//[[Rcpp::export]]
double m0mvnorm_dens(const arma::vec& x, const arma::mat& Si){
  int p = Si.n_cols;
  double normcore =  arma::conv_to<double>::from(x.t() * Si * x);
  double normconst = - p/2.0 * log(2*M_PI) + .5 * log(arma::det(Si));
  return normconst - 0.5 * (normcore);
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
// and gprior for b
//[[Rcpp::export]]
double clm_marglik(const arma::vec& y, const arma::mat& Mi,
                   const arma::mat& Si, double muSimu, double a, double b){
  int p = Si.n_cols;
  int n = y.n_elem;
  double const1 = a * log(b) + lgamma(a + n/2.0) -  n/2.0 * log(2 * M_PI) - lgamma(a);
  double const2 = 0.5 * log(arma::det(Mi)) - 0.5 * log(arma::det(Si));
  
  double normcore = -(a+n/2.0) * log(b + 0.5 * arma::conv_to<double>::from(y.t() * y - muSimu));
  return const1 + const2 + normcore;
}

// log density of mvnormal mean 0 -- only useful in ratios with gpriors
//[[Rcpp::export]]
double m0mvnorm_dens_gratio(double yy, double yPxy, double g, double p){
  return -0.5*p*log(g+1.0) + 0.5*g/(g+1.0) * yPxy - 0.5*yy;
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
//-- only useful in ratios with gpriors
// and gprior for b
//[[Rcpp::export]]
double clm_marglik_gratio(double yy, double yPxy, double g, int n, double p, double a, double b){
  return -0.5*p*log(g+1.0) - (a+n/2.0)*log(b + 0.5*(yy - g/(g+1.0) * yPxy));
}



BayesLMg::BayesLMg(){ 
  
}

BayesLMg::BayesLMg(const arma::vec& yy, const arma::mat& X, double gin, bool sampling=true, bool fixs = false){
  fix_sigma = fixs;
  sampling_mcmc = sampling;
  
  y = yy;
  n = y.n_elem;
  p = X.n_cols;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  g = gin;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  if(sampling_mcmc){
    XtX = X.t() * X;
    
    In = arma::eye(n, n);
    Mi = 1.0/g * XtX + arma::eye(p,p)*1;
    
    inv_var_post = Mi + XtX;
    Sigma = arma::inv_sympd(inv_var_post);
    mu = Sigma * X.t() * ycenter;
    mutSimu = arma::conv_to<double>::from(mu.t()*inv_var_post*mu);
    if(fix_sigma){
      alpha = 0.0;
      beta = 0.0;
      sigmasq = 1.0;
      b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
      reg_mean = icept + X * b;
    } else {
      alpha = 2.1; // parametrization: a = mean^2 / variance + 2
      beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
      alpha_n = alpha + n/2.0;
      beta_n = beta + 0.5*(-mutSimu + yty);
      
      sample_sigmasq();
      sample_beta();
      reg_mean = icept + X * b;
    }
  }
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * hat(X) * ycenter);
  marglik = get_marglik(fixs);
};



void BayesLMg::change_X(const arma::mat& X){
  p = X.n_cols;
  
  if(sampling_mcmc){
    XtX = X.t() * X;
    Mi = 1.0/g * XtX + arma::eye(p,p)*1;
    inv_var_post = Mi + XtX;
    Sigma = arma::inv_sympd(inv_var_post);
    mu = Sigma * X.t() * ycenter;
    mutSimu = arma::conv_to<double>::from(mu.t()*inv_var_post*mu);
    if(fix_sigma){
      alpha = 0.0;
      beta = 0.0;
      sigmasq = 1.0;
      b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
      reg_mean = icept + X * b;
    } else {
      alpha = 2.1; // parametrization: a = mean^2 / variance + 2
      beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
      alpha_n = alpha + n/2.0;
      beta_n = beta + 0.5*(-mutSimu + yty);
      
      sample_sigmasq();
      sample_beta();
      reg_mean = icept + X * b;
    }
  }
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * hat(X) * ycenter);
  //clog << yPxy << endl;
  marglik = get_marglik(fix_sigma);
}

void BayesLMg::sample_beta(){
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
}

void BayesLMg::sample_sigmasq(){
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
}


BayesSelect::BayesSelect(){ 
  
}

double BayesSelect::get_marglik(bool fix_sigma=false){
  if(fix_sigma){
    return m0mvnorm_dens_gratio(yty, yPxy, g, p);
    //clog << marglik << endl;
  } else {
    return clm_marglik_gratio(yty, yPxy, g, n, p, alpha, beta);
    //clog << marglik << endl;
  }
}

BayesSelect::BayesSelect(const arma::vec& yy, const arma::mat& X, double gin, bool fixs = false){
  fix_sigma = fixs;
  
  alpha = 2.1;
  beta = 1.1;
  
  y = yy;
  n = y.n_elem;
  p = X.n_cols;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  g = gin;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * hat(X) * ycenter);
  marglik = get_marglik(fix_sigma);
};



void BayesSelect::change_X(const arma::mat& X){
  p = X.n_cols;
  
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * hat(X) * ycenter);
  //clog << yPxy << endl;
  marglik = get_marglik(fix_sigma);
}

double BayesLMg::get_marglik(bool fix_sigma=false){
  if(fix_sigma){
    return m0mvnorm_dens_gratio(yty, yPxy, g, p);
    //clog << marglik << endl;
  } else {
    return clm_marglik_gratio(yty, yPxy, g, n, p, alpha, beta);
    //clog << marglik << endl;
  }
}

//[[Rcpp::export]]
Rcpp::List lm_gprior(const arma::vec& y, const arma::mat& X, double g=-1.0, bool fixs=false){
  BayesLMg model;
  if(g==-1.0){
    model = BayesLMg(y, X, y.n_elem, fixs);
  } else {
    model = BayesLMg(y, X, g, fixs);
  }
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = model.b,
    Rcpp::Named("mu") = model.mu,
    Rcpp::Named("sigmasq") = model.sigmasq,
    Rcpp::Named("marglik") = model.marglik
  );
}

//[[Rcpp::export]]
bool boolbern(double p){
  double run = arma::randu();
  return run < p;
}

VarSelMCMC::VarSelMCMC(const arma::vec& yy, const arma::mat& XX, const arma::vec& prior,
                       double gin=-1.0, double model_prior_par=1, bool fixsigma=false, int iter=1){
  //clog << "creating " << endl;
  y = yy;
  X = XX;
  mcmc = iter;
  
  p = X.n_cols;
  arma::vec p_indices = arma::linspace<arma::vec>(0, p-1, p);
  n = y.n_elem;
  
  beta_stored = arma::zeros(p, mcmc);
  gamma_stored = arma::zeros(p, mcmc);
  sigmasq_stored = arma::zeros(mcmc);
  
  gamma = arma::uvec(p);
  for(int j=0; j<p; j++){
    gamma(j) = 1*boolbern(prior(j));
  }
  
  arma::uvec gammaix = arma::find(gamma);
  //clog << "test  1" << endl;
  model = BayesSelect(y, X.cols(gammaix), gin, fixsigma);
  sampling_order = rndpp_shuffle(p_indices);
  //clog << "test  2" << endl;
  for(int m=0; m<mcmc; m++){
    //clog << "order " << sampling_order.t() << endl;
    for(int j=0; j<p; j++){
      int ix = sampling_order(j);
      gamma_proposal = gamma;
      gamma_proposal(ix) = 1-gamma(ix);
      arma::uvec gammaix_proposal = arma::find(gamma_proposal);
      //clog << "proposal gamma " << gamma_proposal << endl;
      model_proposal = model;
      model_proposal.change_X(X.cols(gammaix_proposal));
      //clog << "test  mcmc j " << j << endl;
      double accept_probability = exp(model_proposal.marglik - model.marglik) *
        exp(model_prior_par * (model.p - model_proposal.p));
      
      /*clog << "change to " << model_proposal.p << " from " << model.p << 
       " mlik_ratio " << exp(model_proposal.marglik - model.marglik) << 
       " mlik prop " << exp(model_proposal.marglik) <<
       " mlik orig " << exp(model.marglik) << endl;
       */
      
      int accepted = rndpp_bern(accept_probability);
      if(accepted == 1){
        //clog << "accepted." << endl;
        model = model_proposal;
        gamma = gamma_proposal;
        gammaix = gammaix_proposal;
      }
    }
    
    sampled_model = BayesLMg(y, X.cols(gammaix), gin, true, fixsigma);
    arma::vec beta_full = arma::zeros(p);
    beta_full.elem(gammaix) = sampled_model.b;
    
    beta_stored.col(m) = beta_full;
    gamma_stored.col(m) = arma::conv_to<arma::vec>::from(gamma);
    sigmasq_stored(m) = sampled_model.sigmasq;
  }
  
}

//[[Rcpp::export]]
Rcpp::List bvs(const arma::vec& y, const arma::mat& X, 
               const arma::vec& prior, int mcmc,
               double g=-1.0, double model_prior_par=1.0, bool fixs=false){
  
  if(g==-1){
    g = y.n_elem;
  }
  VarSelMCMC bvs_model(y, X, prior, g, model_prior_par, fixs, mcmc);
  return Rcpp::List::create(
    Rcpp::Named("beta") = bvs_model.beta_stored,
    //Rcpp::Named("mu") = bvs_model.mu,
    //Rcpp::Named("sigmasq") = bvs_model.sigmasq,
    Rcpp::Named("gamma") = bvs_model.gamma_stored,
    Rcpp::Named("sigmasq") = bvs_model.sigmasq_stored
  );
}
