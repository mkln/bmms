#include "linear_conjugate.h"

using namespace std;
using namespace Rcpp;


arma::mat rndpp_stdmvnormal(int n, int dimension){
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  for(int i=0; i<n; i++){
    xtemp = Rcpp::rnorm(dimension, 0.0, 1.0);
    outmat.row(i) = xtemp.t();
  }
  return outmat;
}


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
  
  Ip = arma::eye(p,p);
  m = arma::zeros(p);
  //M = n*XtXi;
  //clog << lambda << endl;
  Mi = pow(1.0/n, 0.5) * XtX + Ip * lambda;
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
    b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
    //b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
    
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
  //M = Ip * lambda_new;
  Mi = pow(1.0/n, 0.5) * XtX + Ip * lambda_new;
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  
  //alpha = 2.25; // parametrization: a = mean^2 / variance + 2
  //beta = 0.625; //alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  alpha_n = alpha + n/2.0;
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
  
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  reg_mean = X * b;
}

void BayesLM::posterior(){
  sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
}

void BayesLM::beta_sample(){
  b = (rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  //b = rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
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


arma::mat hat_alt(const arma::mat& X){
  arma::mat U;
  arma::vec s;
  arma::mat V;
  int c = X.n_cols > X.n_rows ? X.n_rows : X.n_cols;
  arma::svd(U,s,V,X);
  return U.cols(0,c-1) * U.cols(0,c-1).t();
}


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
double m0mvnorm_dens(const arma::vec& x, const arma::mat& Si){
  int p = Si.n_cols;
  double normcore =  arma::conv_to<double>::from(x.t() * Si * x);
  double normconst = - p/2.0 * log(2*M_PI) + .5 * log(arma::det(Si));
  return normconst - 0.5 * (normcore);
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
// and gprior for b
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
double m0mvnorm_dens_gratio(double yy, double yPxy, double g, double p){
  return -0.5*p*log(g+1.0) + 0.5*g/(g+1.0) * yPxy - 0.5*yy;
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
//-- only useful in ratios with gpriors
// and gprior for b
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

double BayesLMg::get_marglik(bool fix_sigma=false){
  if(fix_sigma){
    return m0mvnorm_dens_gratio(yty, yPxy, g, p);
    //clog << marglik << endl;
  } else {
    return clm_marglik_gratio(yty, yPxy, g, n, p, alpha, beta);
    //clog << marglik << endl;
  }
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

//' A simple Bayesian Variable Selection model using g-priors
//' 
//' @param y vector of responses
//' @param X design matrix
//' @param prior starting model (#name to be changed#)
//' @param mcmc number of Markov chain Monte Carlo iterations
//' @param g g-prior parameter
//' @param model_prior_par For model M, p(M) is prop. to exp(k * p) where p is the 
//'   number of included variables, and k is model_prior_par
//' @param fixs Fix the regression variance to 1?
//' @export
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


ModularVS::ModularVS(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, 
                     const arma::field<arma::vec>& starting,
                     int mcmc_in,
                     double gg, arma::vec module_prior_par, bool binary=false){
  
  K = Xall_in.n_elem;
  //clog << K << endl;
  mcmc = mcmc_in;
  y = y_in;
  Xall = Xall_in;
  resid = y;
  
  n = y.n_elem;
  arma::vec z(n);
  arma::mat zsave(n, mcmc);
  
  z = (y-0.5)*2;
  
  arma::uvec yones = arma::find(y == 1);
  arma::uvec yzeros = arma::find(y == 0);
  
  // y=1 is truncated lower by 0. upper=inf
  // y=0 is truncated upper by 0, lower=-inf
  
  arma::vec trunc_lowerlim = arma::zeros(n);
  trunc_lowerlim.elem(yones).fill(0.0);
  trunc_lowerlim.elem(yzeros).fill(-arma::datum::inf);
  
  arma::vec trunc_upperlim = arma::zeros(n);
  trunc_upperlim.elem(yones).fill(arma::datum::inf);
  trunc_upperlim.elem(yzeros).fill(0.0);
  
  arma::mat In = arma::eye(n,n);
  
  intercept = arma::zeros(K, mcmc);
  beta_store = arma::field<arma::mat>(K);
  gamma_store = arma::field<arma::mat>(K);
  gamma_start = starting;
  
  for(int j=0; j<K; j++){
    gamma_start(j) = arma::zeros(Xall(j).n_cols);
    for(unsigned int h=0; h<Xall(j).n_cols; h++){
      gamma_start(j)(h) = rndpp_bern(0.1);
    }
    beta_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
    gamma_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
  }
  
  for(int m=0; m<mcmc; m++){
    //clog << "m: " << m << endl;
    if(binary){
      resid = z;
    } else {
      resid = y;
    }
    
    arma::vec xb_cumul = arma::zeros(n);
    for(int j=0; j<K; j++){
      //clog << " j: " << j << " " << arma::size(Xall(j)) << endl;
      //clog << "  module" << endl;
      double iboh = arma::mean(resid);
      
      //VSModule last_split_model = VSModule(y, X, gamma_start, MCMC, g, sprior, fixsigma, binary);
      //VarSelMCMC bvs_model(y, X, gamma_start, g, sprior, fixsigma, MCMC);
      
      //VSModule onemodule = VSModule(resid, Xall(j), gamma_start(j), 1, gg, ss(j), binary?true:false, false);
      VarSelMCMC onemodule(resid, Xall(j), gamma_start(j), gg, module_prior_par(j), binary?true:false, 1);
      
      //varsel_modules.push_back(onemodule);
      intercept(j, m) = iboh;// onemodule.intercept;
      xb_cumul = xb_cumul + Xall(j) * onemodule.beta_stored.col(0);
      resid = resid - xb_cumul;
      //clog << "  beta store" << endl;
      beta_store(j).col(m) = onemodule.beta_stored.col(0);
      //clog << "  gamma store" << endl;
      gamma_store(j).col(m) = onemodule.gamma_stored.col(0);
      //clog << "  gamma start" << endl;
      gamma_start(j) = onemodule.gamma_stored.col(0);
      //clog << gamma_start(1) << endl;
    }
    
    if(binary){
      z = mvtruncnormal_eye1(xb_cumul, 
                             trunc_lowerlim, trunc_upperlim).col(0);
    }
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        clog << m << " " << max(abs(z)) << endl;
      } 
    }
  }
}

//' A Modular & Multiscale Bayesian Variable Selection model
//' 
//' @param y vector of responses
//' @param Xall A list of length K, where each component is a design matrix at a different resolution.
//' @param starting A list of length K with the starting configurations. Useful to restart MCMC from good locations
//' @param mcmc_in number of Markov chain Monte Carlo iterations
//' @param gg g-prior parameter
//' @param module_prior_par A vector with K components.  For model M, p(M) is prop. to exp(k * p) where p is the 
//'   number of included variables, and k is model_prior_par
//' @param binary Are responses binary (0,1)?
//' @export
// [[Rcpp::export]]
Rcpp::List momscaleBVS(const arma::vec& y, 
                            const arma::field<arma::mat>& Xall, 
                            const arma::field<arma::vec>& starting,
                            int mcmc, double gg, arma::vec module_prior_par, bool binary=false){
  
  int n = y.n_elem;
  ModularVS test = ModularVS(y, Xall, starting, mcmc, gg, module_prior_par, binary);
  
  return Rcpp::List::create(
    Rcpp::Named("intercept_mc") = test.intercept,
    Rcpp::Named("beta_mc") = test.beta_store,
    Rcpp::Named("gamma_mc") = test.gamma_store
  );
}
