//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef image_voronoi
#define image_voronoi

//#include <RcppArmadillo.h>
#include "metrop_helper.h"

inline double blm_marglik(arma::vec& y, arma::mat& mean_post, arma::mat& inv_var_post, double a, double b){
  int p1 = inv_var_post.n_cols;
  int n = y.n_elem;
  double result = (n-p1)*log(n*1.0) - (a+n/2.0) * log(b + .5*arma::conv_to<double>::from(y.t()*y - mean_post.t() * inv_var_post * mean_post));
  return result;
}

class BayesLM2D{
public:
  
  bool fix_sigma;
  
  arma::mat X_flat;
  arma::cube Xcube;
  
  arma::vec y;
  
  arma::mat In;
  
  int p1, p2;
  
  arma::field<arma::mat> splitmat;
  arma::mat splitmask;
  
  arma::mat Beta;
  arma::mat groupmask;
  arma::vec regions;
  
  double lambda;
  
  bmmodels::BayesLM flatmodel; //***
  //VSModule selector;
  
  arma::mat mask_nosplits;
  
  //arma::mat mu_post;
  //arma::mat Sigma_post;
  double sigmasq_post_mean;
  double a_post;
  double b_post;
  
  double icept_sampled;
  arma::mat beta_sampled;
  double sigmasq_sampled;
  
  int effective_dimension;
  
  double logmarglik();
  
  void change_splits(arma::field<arma::mat>&);
  //void change_data(const arma::vec&);
  
  arma::vec residuals; 
  arma::vec Xb;
  
  BayesLM2D(arma::vec&, arma::cube&, arma::field<arma::mat>&, arma::mat&, double, bool, double);
};

inline BayesLM2D::BayesLM2D(arma::vec& yin, arma::cube& X2d, 
                     arma::field<arma::mat>& splits, arma::mat& mask_forbid, 
                     double lambda_in, bool fixed_sigmasq, double sigmasqin = -1.0){
  
  //cout << "[BayesLM2D] starting setup" << endl; 
  fix_sigma = fixed_sigmasq;
  
  if(fix_sigma & (sigmasqin==-1)){
    sigmasq_sampled = 1.0;
  } else {
    if(fix_sigma){
      sigmasq_sampled = sigmasqin; //
    } else {
      sigmasq_sampled = -1.0; // will be sampled by BayesLM
    }
  }
  
  p1 = X2d.n_rows;
  p2 = X2d.n_cols;
  
  lambda = lambda_in;
  
  y = yin;
  In = arma::eye(y.n_elem, y.n_elem);
  
  splitmat = splits;
  splitmask = bm2d::splitsub_to_splitmask(splitmat, p1, p2);
  mask_nosplits = mask_forbid;
  
  groupmask = bm2d::splitsub_to_groupmask(splits, p1, p2);
  regions = bm2d::mat_unique(groupmask);
  Xcube = X2d;
  X_flat = bm2d::cube_to_mat_by_region(X2d, groupmask, regions);
  //clog << col_sums(X_flat) << endl;
  effective_dimension = X_flat.n_cols;
  
  //cout << "[BayesLM2D] fitting flat model" << endl;
  //cout << fix_sigma << " " << sigmasq_sampled << endl;
  flatmodel = bmmodels::BayesLM(y, X_flat, lambda, fix_sigma, sigmasq_sampled); 
  //flatmodel = VSModule(y, X_flat, lambda, fix_sigma);
  
  //cout << "[BayesLM2D] saving" << endl;
  if(!fix_sigma){
    sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
    a_post = flatmodel.alpha_n;
    b_post = flatmodel.beta_n;
  } else {
    sigmasq_post_mean = sigmasq_sampled;
    a_post = -1;
    b_post = -1;
  }
  
  beta_sampled = bm2d::unmask_vector(flatmodel.b, regions, groupmask);
  sigmasq_sampled = flatmodel.sigmasq;
  
  Xb = flatmodel.reg_mean;
  residuals = y - Xb;
  icept_sampled = arma::mean(residuals);
  //clog << "created BLM2D" << endl;
}

inline void BayesLM2D::change_splits(arma::field<arma::mat>& splits){
  splitmat = splits;
  splitmask = bm2d::splitsub_to_splitmask(splitmat, p1, p2);
  
  groupmask = bm2d::splitsub_to_groupmask(splits, p1, p2);
  regions = bm2d::mat_unique(groupmask);
  
  X_flat = bm2d::cube_to_mat_by_region(Xcube, groupmask, regions);
  effective_dimension = X_flat.n_cols;
  
  flatmodel = bmmodels::BayesLM(y, X_flat, lambda, fix_sigma, sigmasq_sampled); 
  //flatmodel = VSModule(y, X_flat, lambda, fix_sigma);
  
  if(!fix_sigma){
    sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
    a_post = flatmodel.alpha_n;
    b_post = flatmodel.beta_n;
  } else {
    sigmasq_post_mean = sigmasq_sampled;
    a_post = -1;
    b_post = -1;
  }
  
  //icept_sampled = flatmodel.icept;
  beta_sampled = bm2d::unmask_vector(flatmodel.b, regions, groupmask);
  sigmasq_sampled = flatmodel.sigmasq;
  
  Xb = flatmodel.reg_mean;
  residuals = y - Xb;
  icept_sampled = arma::mean(residuals);
}

inline double BayesLM2D::logmarglik(){
  if(fix_sigma){
    //cout << "[BayesLM2D::logmarglik] calculating" << endl;
    double tt = modular_loglikn(flatmodel.y - flatmodel.reg_mean, 1.0/flatmodel.sigmasq * (In - flatmodel.Px));
    //cout << "[BayesLM2D::logmarglik] done " << endl;
    return tt;
  } else {
    return blm_marglik(flatmodel.ycenter, flatmodel.mu, flatmodel.inv_var_post, 
                       flatmodel.alpha, flatmodel.beta);
  }
}

class ModularLR2D {
public:
  
  bool fix_sigma;
  
  // input data
  int n;
  int p1, p2;
  
  double lambda; // ridge
  
  arma::vec y;
  arma::cube X2d;
  
  // limit to number of possible stages
  int max_stages;
  
  // grid splits
  int n_stages;
  
  arma::vec logliks;
  
  arma::field<arma::mat> splitsub;
  std::vector<BayesLM2D> modules;
  
  void change_module(int, arma::field<arma::mat>&);
  //void change_all(arma::field<arma::mat>&);
  void add_new_module(const arma::field<arma::mat>&);
  void delete_last_module();
  //void change_data(const arma::vec&);
  
  arma::vec ones;
  arma::vec intercept;
  arma::vec icept_sampled;
  arma::vec sigmasq_sampled;
  arma::cube theta_sampled;
  arma::mat mask_nosplits;
  
  arma::vec Xb_sum;
  //void redo();
  ModularLR2D();
  ModularLR2D(const arma::vec&, const arma::cube&, const arma::field<arma::mat>&, arma::mat&, int, double);
  ModularLR2D(const arma::vec&, const arma::cube&, const arma::field<arma::mat>&, arma::mat&, int, double, 
              bool, double);
};

inline ModularLR2D::ModularLR2D(const arma::vec& yin, const arma::cube& Xin, const arma::field<arma::mat>& in_splits, 
                         arma::mat& mask_forbid, int set_max_stages=5, double lambda_in = 0.5){
  
  fix_sigma = false;
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
  sigmasq_sampled = arma::zeros(max_stages);
  
  double sigprior = 0;
  
  for(unsigned int s = 0; s<n_stages; s++){
    //clog << s << endl;
    arma::field<arma::mat> current_split_field(s+1);
    for(unsigned int j=0; j<s+1; j++){
      current_split_field(j) = in_splits(j);
    }
    BayesLM2D adding_module(residuals, X2d, current_split_field, mask_nosplits, lambda, fix_sigma);
    modules.push_back(adding_module);
    logliks(s) = adding_module.logmarglik() + sigprior;
    theta_sampled.slice(s) = adding_module.beta_sampled;
    icept_sampled(s) = adding_module.icept_sampled;
    sigmasq_sampled(s) = adding_module.sigmasq_sampled;
    residuals = adding_module.residuals;
    if(!fix_sigma){
      sigprior += R::dgamma(1.0/adding_module.sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1);//***
    }
    Xb_sum += adding_module.Xb;
  }
}

inline ModularLR2D::ModularLR2D(){
  
}


inline ModularLR2D::ModularLR2D(const arma::vec& yin, const arma::cube& Xin, const arma::field<arma::mat>& in_splits, 
                         arma::mat& mask_forbid, int set_max_stages=5, double lambda_in = 0.5, 
                         bool fixed_sigma=false, double sigmasqin=-1.0){
  
  max_stages = set_max_stages;
  fix_sigma = fixed_sigma;
  if(fix_sigma & (sigmasqin==-1)){
    sigmasq_sampled = arma::ones(max_stages);
  } else {
    if(fix_sigma){
      sigmasq_sampled = arma::ones(max_stages) * sigmasqin; //
    } else {
      sigmasq_sampled = arma::zeros(max_stages)-1; // sigmas will be set later
    }
  }
  
  //clog << "[ModularLR2D] sigmasq sampled " << sigmasq_sampled.t() << endl;
  
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
  
  
  icept_sampled = arma::zeros(max_stages);
  theta_sampled = arma::zeros(p1, p2, max_stages);
  
  double sigprior = 0;
  
  //cout << "[ModularLR2D] setting up the modules" << endl;
  for(unsigned int s = 0; s<n_stages; s++){
    //clog << "[ModularLR2D] module " << s << endl;
    arma::field<arma::mat> current_split_field(s+1);
    for(unsigned int j=0; j<s+1; j++){
      current_split_field(j) = in_splits(j);
    }
    BayesLM2D adding_module(residuals, X2d, 
                            current_split_field, mask_nosplits, lambda, 
                            fix_sigma, sigmasq_sampled(s));
    modules.push_back(adding_module);
    logliks(s) = adding_module.logmarglik() + sigprior;
    //clog << "v1" << endl;
    icept_sampled(s) = adding_module.icept_sampled;
    //clog << "v2" << endl;
    theta_sampled.slice(s) = adding_module.beta_sampled;
    sigmasq_sampled(s) = adding_module.sigmasq_sampled;
    residuals = adding_module.residuals;
    if(!fix_sigma) {
      sigprior += R::dgamma(1.0/adding_module.sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
    }
    Xb_sum += adding_module.Xb;
    //cout << "[ModularLR2D] module " << s << " done." << endl;
  }
}

inline void ModularLR2D::change_module(int which_level, arma::field<arma::mat>& in_splits){
  splitsub = in_splits;
  
  arma::vec residuals;
  if(which_level > 0){
    residuals = modules[which_level-1].residuals;
  } else {
    residuals = y;
  }
  
  double sigprior = 0.0;
  if(!fix_sigma){
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
    sigmasq_sampled(s) = modules[s].sigmasq_sampled;
    residuals = modules[s].residuals;
    if(!fix_sigma){
      sigprior += R::dgamma(1.0/modules[s].sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
    }
  }
  
  Xb_sum = arma::zeros(n);
  for(unsigned int s = 0; s<n_stages; s++){
    Xb_sum += modules[s].Xb;
  }
}

inline void ModularLR2D::add_new_module(const arma::field<arma::mat>& in_splits){
  splitsub = in_splits;
  int s = n_stages;
  arma::field<arma::mat> current_split_field(s+1);
  for(unsigned int j=0; j<s+1; j++){
    current_split_field(j) = in_splits(j);
  }
  BayesLM2D adding_module(y, X2d, current_split_field, mask_nosplits, lambda, fix_sigma);
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
  sigmasq_sampled(n_stages-1) = adding_module.sigmasq_sampled;
  
  Xb_sum = arma::zeros(n);
  for(unsigned int s = 0; s<n_stages; s++){
    Xb_sum += modules[s].Xb;
  }
}

inline void ModularLR2D::delete_last_module(){
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

inline double gammaprior_mhr(double new_val, double old_val, double alpha=100, double beta=.5){
  return (alpha-1) * (log(new_val) - log(old_val)) - 1.0/beta * (new_val - old_val);
}




arma::field<arma::mat> merge_splits(arma::field<arma::mat>& old_splits, arma::field<arma::mat> new_splits);



inline double struct2d_prior_ratio(const arma::field<arma::vec>& proposed, const arma::field<arma::vec>& original,
                            int stage, int p, double param){
  int stage_proposed=stage, stage_original=stage;
  if(stage==-1){
    stage_proposed = proposed.n_elem-1;
    stage_original = original.n_elem-1;
    while(proposed(stage_proposed).n_elem==0){
      stage_proposed--;
    }
    while(original(stage_original).n_elem==0){
      stage_original--;
    }
    
  }
  arma::vec proposed_here_diff = arma::diff(arma::join_vert(arma::sort(proposed(stage_proposed)), arma::ones(1)*p));
  
  arma::vec original_here_diff = arma::diff(arma::join_vert(arma::sort(original(stage_original)), arma::ones(1)*p));
  
  double rat = 1.0;
  try{
    double minprop = proposed_here_diff.min();
    if(minprop == 0){
      clog << "Runtime error: diff in splits=0 hence wrong move." << endl;
      throw 1;
    }
    double minorig = original_here_diff.min();
    if(minorig == 0){
      clog << "Runtime error: diff in splits=0 hence wrong move." << endl;
      throw 1;
    }
    rat = pow(log(1.0+minprop)/log(1.0+minorig), param);
    //clog << "ratio " << rat << endl;
    return rat;
  } catch(...) {
    // min has no elements error -- happens when proposing drop from 2 to 1.
    return 1.0;
  }
  
}


#endif