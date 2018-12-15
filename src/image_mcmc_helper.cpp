//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]


#include "image_mcmc_helper.h"

using namespace std;

#include <string>

//[[Rcpp::export]]
arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m){
  return bmdataman::index_to_subscript(index, m);
}

double blm_marglik(arma::vec& y, arma::mat& mean_post, arma::mat& inv_var_post, double a, double b){
  int p1 = inv_var_post.n_cols;
  int n = y.n_elem;
  double result = (n-p1)*log(n*1.0) - (a+n/2.0) * log(b + .5*arma::conv_to<double>::from(y.t()*y - mean_post.t() * inv_var_post * mean_post));
  return result;
}

/*
 double blm_marglik(arma::vec y, arma::mat& m, arma::mat& M, double a=2.1, double b=1.1, bool fixsigma=true){
 double result;
 if(fixsigma){ // sigma fixed to 1
 int n = y.n_elem;
 // m = mean_pre = zero
 // M = inv_var_pre
 result = -n*log(2*M_PI)-.5*arma::conv_to<double>::from( y.t() * M * y);
 } else {
 
 // m = mean_post
 // M = inv_var_post
 int p1 = M.n_cols;
 int n = y.n_elem;
 result = (n-p1)*log(n*1.0) - (a+n/2.0) * log(b + .5*arma::conv_to<double>::from(y.t()*y - m.t() * M * m));
 }
 return result;
 }
 */

BayesLM2D::BayesLM2D(arma::vec& yin, arma::cube& X2d, 
                     arma::field<arma::mat>& splits, arma::mat& mask_forbid, 
                     double lambda_in, bool biny){
  binary = biny;
  
  p1 = X2d.n_rows;
  p2 = X2d.n_cols;
  
  lambda = lambda_in;
  
  y = yin;
  
  splitmat = splits;
  splitmask = bm2d::splitsub_to_splitmask(splitmat, p1, p2);
  mask_nosplits = mask_forbid;
  
  groupmask = bm2d::splitsub_to_groupmask(splits, p1, p2);
  regions = bm2d::mat_unique(groupmask);
  Xcube = X2d;
  X_flat = bm2d::cube_to_mat_by_region(X2d, groupmask, regions);
  //clog << col_sums(X_flat) << endl;
  effective_dimension = X_flat.n_cols;
  flatmodel = bmmodels::BayesLM(y, X_flat, lambda, binary); 
  //flatmodel = VSModule(y, X_flat, lambda, binary);
  
  
  //mu_post = unmask_vector(flatmodel.mu, regions, groupmask);
  //Sigma_post = flatmodel.Sigma;
  sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
  a_post = flatmodel.alpha_n;
  b_post = flatmodel.beta_n;
  
  icept_sampled = flatmodel.icept;
  beta_sampled = bm2d::unmask_vector(flatmodel.b, regions, groupmask);
  sigmasq_sampled = flatmodel.sigmasq;
  
  Xb = flatmodel.reg_mean;
  residuals = y - Xb;
  //clog << "created BLM2D" << endl;
}

void BayesLM2D::change_splits(arma::field<arma::mat>& splits){
  splitmat = splits;
  splitmask = bm2d::splitsub_to_splitmask(splitmat, p1, p2);
  
  groupmask = bm2d::splitsub_to_groupmask(splits, p1, p2);
  regions = bm2d::mat_unique(groupmask);
  
  X_flat = bm2d::cube_to_mat_by_region(Xcube, groupmask, regions);
  effective_dimension = X_flat.n_cols;
  
  flatmodel = bmmodels::BayesLM(y, X_flat, lambda, binary); 
  //flatmodel = VSModule(y, X_flat, lambda, binary);
  
  //mu_post = unmask_vector(flatmodel.mu, regions, groupmask);
  //Sigma_post = flatmodel.Sigma;
  sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
  a_post = flatmodel.alpha_n;
  b_post = flatmodel.beta_n;
  
  icept_sampled = flatmodel.icept;
  beta_sampled = bm2d::unmask_vector(flatmodel.b, regions, groupmask);
  sigmasq_sampled = flatmodel.sigmasq;
  
  Xb = flatmodel.reg_mean;
  residuals = y - Xb;
}


double BayesLM2D::logmarglik(){
  return blm_marglik(flatmodel.ycenter, flatmodel.mu, flatmodel.inv_var_post, 
                     flatmodel.alpha, flatmodel.beta);
}

/*****
 double BayesLM2D::logmarglik(){
 return blm_marglik(flatmodel.y - flatmodel.icept, flatmodel.mu, flatmodel.inv_mlik_var, 
 flatmodel.alpha, flatmodel.betas, true);
 }
 */

ModularLR2D::ModularLR2D(const arma::vec& yin, const arma::cube& Xin, const arma::field<arma::mat>& in_splits, 
                         arma::mat& mask_forbid, int set_max_stages=5, double lambda_in = 0.5){
  
  binary = false;
  splitsub = in_splits;
  p1 = Xin.n_rows;
  p2 = Xin.n_cols;
  n = yin.n_elem;
  
  y = yin;
  X2d = Xin;
  
  lambda = lambda_in;
  
  mask_nosplits = mask_forbid;
  
  n_stages = in_splits.n_elem;
  logliks = arma::zeros(n_stages);
  arma::vec residuals = y;
  Xb_sum = arma::zeros(n);
  
  max_stages = set_max_stages;
  icept_sampled = arma::zeros(max_stages);
  theta_sampled = arma::zeros(p1, p2, max_stages);
  double sigprior = 0;
  
  for(unsigned int s = 0; s<n_stages; s++){
    //clog << s << endl;
    arma::field<arma::mat> current_split_field(s+1);
    for(unsigned int j=0; j<s+1; j++){
      current_split_field(j) = in_splits(j);
    }
    BayesLM2D adding_module(residuals, X2d, current_split_field, mask_nosplits, lambda, binary);
    modules.push_back(adding_module);
    logliks(s) = adding_module.logmarglik() + sigprior;
    theta_sampled.slice(s) = adding_module.beta_sampled;
    icept_sampled(s) = adding_module.icept_sampled;
    residuals = adding_module.residuals;
    if(!binary){
      sigprior += R::dgamma(1.0/adding_module.sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1);//***
    }
    Xb_sum += adding_module.Xb;
  }
}


ModularLR2D::ModularLR2D(const arma::vec& yin, const arma::cube& Xin, const arma::field<arma::mat>& in_splits, 
                         arma::mat& mask_forbid, int set_max_stages=5, double lambda_in = 0.5, bool biny=false){
  
  binary = biny;
  splitsub = in_splits;
  p1 = Xin.n_rows;
  p2 = Xin.n_cols;
  n = yin.n_elem;
  
  y = yin;
  X2d = Xin;
  
  lambda = lambda_in;
  
  mask_nosplits = mask_forbid;
  
  n_stages = in_splits.n_elem;
  logliks = arma::zeros(n_stages);
  arma::vec residuals = y;
  Xb_sum = arma::zeros(n);
  
  max_stages = set_max_stages;
  icept_sampled = arma::zeros(max_stages);
  theta_sampled = arma::zeros(p1, p2, max_stages);
  double sigprior = 0;
  
  for(unsigned int s = 0; s<n_stages; s++){
    //clog << s << endl;
    arma::field<arma::mat> current_split_field(s+1);
    for(unsigned int j=0; j<s+1; j++){
      current_split_field(j) = in_splits(j);
    }
    BayesLM2D adding_module(residuals, X2d, current_split_field, mask_nosplits, lambda, binary);
    modules.push_back(adding_module);
    logliks(s) = adding_module.logmarglik() + sigprior;
    //clog << "v1" << endl;
    icept_sampled(s) = adding_module.icept_sampled;
    //clog << "v2" << endl;
    theta_sampled.slice(s) = adding_module.beta_sampled;
    residuals = adding_module.residuals;
    if(!binary) {
      sigprior += R::dgamma(1.0/adding_module.sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
    }
    Xb_sum += adding_module.Xb;
  }
}

void ModularLR2D::change_module(int which_level, arma::field<arma::mat>& in_splits){
  splitsub = in_splits;
  
  arma::vec residuals;
  if(which_level > 0){
    residuals = modules[which_level-1].residuals;
  } else {
    residuals = y;
  }
  
  double sigprior = 0.0;
  if(!binary){
    for(unsigned int s = 0; s<which_level; s++){
      sigprior += R::dgamma(1.0/modules[s].sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
    }
  }
  
  for(unsigned int s = which_level; s<n_stages; s++){
    arma::field<arma::mat> current_split_field(s+1);
    for(unsigned int j=0; j<s+1; j++){
      current_split_field(j) = in_splits(j);
    }
    
    //BayesLM2D adding_module(residuals, X2d, current_split_field);
    modules[s].change_splits(current_split_field);
    logliks(s) = modules[s].logmarglik() + sigprior;
    icept_sampled(s) = modules[s].icept_sampled;
    theta_sampled.slice(s) = modules[s].beta_sampled;
    residuals = modules[s].residuals;
    if(!binary){
      sigprior += R::dgamma(1.0/modules[s].sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
    }
  }
  
  Xb_sum = arma::zeros(n);
  for(unsigned int s = 0; s<n_stages; s++){
    Xb_sum += modules[s].Xb;
  }
}

void ModularLR2D::add_new_module(const arma::field<arma::mat>& in_splits){
  splitsub = in_splits;
  int s = n_stages;
  arma::field<arma::mat> current_split_field(s+1);
  for(unsigned int j=0; j<s+1; j++){
    current_split_field(j) = in_splits(j);
  }
  BayesLM2D adding_module(y, X2d, current_split_field, mask_nosplits, lambda, binary);
  modules.push_back(adding_module);
  n_stages ++;
  arma::vec old_logliks = logliks;
  logliks = arma::zeros(n_stages);
  for(unsigned int s=0; s<n_stages-1; s++){
    logliks(s) = old_logliks(s);
  }
  logliks(n_stages-1) = adding_module.logmarglik();
  icept_sampled(n_stages-1) = adding_module.icept_sampled;
  theta_sampled.slice(n_stages-1) = adding_module.beta_sampled;
  
  Xb_sum = arma::zeros(n);
  for(unsigned int s = 0; s<n_stages; s++){
    Xb_sum += modules[s].Xb;
  }
}

void ModularLR2D::delete_last_module(){
  n_stages--;
  arma::vec old_logliks = logliks;
  logliks = arma::zeros(n_stages);
  for(unsigned int s=0; s<n_stages; s++){
    logliks(s) = old_logliks(s);
  }
  
  
  modules.pop_back();
  theta_sampled.slice(n_stages-1) = arma::zeros(p1, p2);
  icept_sampled(n_stages-1) = 0.0;
  
  arma::field<arma::mat> splitsub_new = arma::field<arma::mat>(n_stages);
  Xb_sum = arma::zeros(n);
  for(unsigned int s = 0; s<n_stages; s++){
    Xb_sum += modules[s].Xb;
    splitsub_new(s) = splitsub(s);
  }
  splitsub = splitsub_new;
}

//' @export
//[[Rcpp::export]]
arma::field<arma::mat> load_splits(int maxlevs, std::string sname){
  arma::field<arma::mat> splits(maxlevs);
  splits.load(sname);
  splits.print();
  return splits;
}


double gammaprior_mhr(double new_val, double old_val, double alpha, double beta){
  return (alpha-1) * (log(new_val) - log(old_val)) - 1.0/beta * (new_val - old_val);
}

