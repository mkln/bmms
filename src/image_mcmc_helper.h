//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef image_voronoi
#define image_voronoi

//#include <RcppArmadillo.h>
#include "metrop_helper.h"


inline double logdet(const arma::mat& X){
  double val, sign;
  arma::log_det(val, sign, X);
  return val;
}


// log density of mvnormal mean 0
inline double m0mvnorm_dens(const arma::vec& x, const arma::mat& Si){
  int p = Si.n_cols;
  double normcore =  arma::conv_to<double>::from(x.t() * Si * x);
  double normconst = - p/2.0 * log(2*M_PI) + .5 * logdet(Si);
  return normconst - 0.5 * (normcore);
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
// and gprior for b
inline double clm_marglik(const arma::vec& y, const arma::mat& Mi,
                          const arma::mat& Si, double muSimu, double a, double b){
  int p = Si.n_cols;
  int n = y.n_elem;
  double const1 = a * log(b) + lgamma(a + n/2.0) -  n/2.0 * log(2 * M_PI) - lgamma(a);
  double const2 = 0.5 * logdet(Mi) - 0.5 * logdet(Si);
  
  double normcore = -(a+n/2.0) * log(b + 0.5 * arma::conv_to<double>::from(y.t() * y - muSimu));
  return const1 + const2 + normcore;
}


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
  double radius;
  double g;
  
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
  
  void change_splits(const arma::vec&, const arma::field<arma::mat>&, bool);
  void chg_y(const arma::vec&);
  //void change_data(const arma::vec&);
  
  arma::vec residuals; 
  arma::vec Xb;
  
  BayesLM2D(const arma::vec&, const arma::cube&, const arma::field<arma::mat>&, const arma::mat&, 
            double, bool, double, double, double);
};

inline BayesLM2D::BayesLM2D(const arma::vec& yin, const arma::cube& X2d, 
                            const arma::field<arma::mat>& splits, const arma::mat& mask_forbid, 
                     double lambda_in, bool fixed_sigmasq, double sigmasqin = -1.0, double gin=-1, double radiusin=-1){
  
  //cout << "[BayesLM2D] starting setup" << endl; 
  fix_sigma = fixed_sigmasq;
  sigmasq_sampled = sigmasqin;
  
  //clog << "BLM2D " << fix_sigma << " " << sigmasq_sampled << endl;
  
  p1 = X2d.n_rows;
  p2 = X2d.n_cols;
  
  lambda = lambda_in;
  g = gin;
  radius = radiusin;
  
  y = yin;
  In = arma::eye(y.n_elem, y.n_elem);
  
  splitmat = splits;
  splitmask = bm2d::splitsub_to_splitmask(splitmat, p1, p2);
  mask_nosplits = mask_forbid;
  
  if(radius != -1){
    groupmask = bm2d::splitsub_to_groupmask_bubbles(splits, p1, p2, radius);
    regions = bmfuncs::bmms_setdiff(bm2d::mat_unique(groupmask), arma::zeros(1));
    //clog << regions << endl;
  } else {
    groupmask = bm2d::splitsub_to_groupmask(splits, p1, p2);
    regions = bm2d::mat_unique(groupmask);
  }
  
  Xcube = X2d;
  X_flat = bm2d::cube_to_mat_by_region(X2d, groupmask, regions);
  //clog << col_sums(X_flat) << endl;
  effective_dimension = X_flat.n_cols;
  
  //clog << "prep lm" << endl;
  flatmodel = bmmodels::BayesLM(y, X_flat, arma::zeros(X_flat.n_cols), 
                                                             .1, .1, 
                                lambda, fix_sigma, sigmasq_sampled, g); 

  if(!fix_sigma){
    sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
    a_post = flatmodel.alpha_n;
    b_post = flatmodel.beta_n;
  } else {
    sigmasq_post_mean = sigmasq_sampled;
    a_post = -1;
    b_post = -1;
  }
  //clog << "done lm. prep beta" << endl; 
  beta_sampled = bm2d::unmask_vector(flatmodel.b, regions, groupmask);
  //clog << "done beta" << endl;
  /*clog << "BayesLM2D >> s: in=" << sigmasqin << 
    ", mid=" << sigmasq_sampled << 
      ", out=" << flatmodel.sigmasq << 
        " logpost:" << flatmodel.logpost << 
          " fixed? " << fix_sigma << endl;
  */
  
  sigmasq_sampled = flatmodel.sigmasq;
  
  Xb = flatmodel.reg_mean;
  residuals = y - Xb;
  icept_sampled = arma::mean(residuals);
  }

inline void BayesLM2D::change_splits(const arma::vec& yin, const arma::field<arma::mat>& splits, bool resample_sigmasq=true){
  splitmat = splits;
  splitmask = bm2d::splitsub_to_splitmask(splitmat, p1, p2);
  
  y = yin;
  
  if(radius != -1){
    groupmask = bm2d::splitsub_to_groupmask_bubbles(splits, p1, p2, radius);
    regions = bmfuncs::bmms_setdiff(bm2d::mat_unique(groupmask), arma::zeros(1));
    //clog << regions << endl;
  } else {
    groupmask = bm2d::splitsub_to_groupmask(splits, p1, p2);
    regions = bm2d::mat_unique(groupmask);
  }
  
  X_flat = bm2d::cube_to_mat_by_region(Xcube, groupmask, regions);
  effective_dimension = X_flat.n_cols;
  
  flatmodel = bmmodels::BayesLM(y, X_flat, arma::zeros(X_flat.n_cols), 
                                                             .1, .1, 
                                lambda, !resample_sigmasq,//fix_sigma, 
                                sigmasq_sampled, g); 
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

inline void BayesLM2D::chg_y(const arma::vec& yin){
  y = yin;
  
  flatmodel.chg_y(y, fix_sigma);
  
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
}

inline double BayesLM2D::logmarglik(){
  return flatmodel.logpost;
}

class ModularLR2D {
public:
  
  bool binary_outcome;
  bool fix_sigma;
  
  // input data
  int n;
  int p1, p2;
  
  double lambda; // ridge
  double radius;
  double g;
  
  arma::vec y;
  arma::cube X2d;
  
  // limit to number of possible stages
  int max_stages;
  
  // grid splits
  int n_stages;
  
  arma::vec logliks;
  
  arma::field<arma::mat> splitsub;
  std::vector<BayesLM2D> modules;
  
  void change_module(int, const arma::field<arma::mat>&);
  void propose_change_module(int, const arma::field<arma::mat>&, bool);
  void confirm_change_module(int, const arma::field<arma::mat>&, bool);
  
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
              bool, bool, double, double, double);
};

inline ModularLR2D::ModularLR2D(const arma::vec& yin, const arma::cube& Xin, const arma::field<arma::mat>& in_splits, 
                         arma::mat& mask_forbid, int set_max_stages=5, double lambda_in = 0.5){
  
  fix_sigma = false;
  binary_outcome = false;
  splitsub = in_splits;
  p1 = Xin.n_rows;
  p2 = Xin.n_cols;
  n = yin.n_elem;
  
  y = yin;
  X2d = Xin;
  
  lambda = lambda_in;
  radius = -1.0;
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
    BayesLM2D adding_module(residuals, X2d, current_split_field, mask_nosplits, lambda, fix_sigma, radius);
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
                         bool fixed_sigma=false, bool binary=false,
                         double sigmasqin=-1.0, double gin = -1, double radiusin=-1.0){
  
  max_stages = set_max_stages;
  
  // sfixed: true, bin: true should be disallowed
  // sfixed: false, bin: true redundant
  // sfixed: true, bin: false advised against
  // sfixed: false, bin: false OK
  // sigmasqin only applies if sfixed=true
  // if binary outcome, then sigma should be fixed at last module, free on the others.
  binary_outcome = binary;
  fix_sigma = fixed_sigma;
  if(binary_outcome){
    if(fix_sigma){
      sigmasq_sampled = arma::ones(max_stages);
    } else {
      sigmasq_sampled = arma::zeros(max_stages)-1;
      sigmasq_sampled(max_stages-1) = 1.0;
    }
  } else {
    if(fix_sigma){
      sigmasq_sampled = arma::ones(max_stages)*sigmasqin;
    } else {
      sigmasq_sampled = arma::zeros(max_stages)-1;
    }
  }
  
  //clog << "MLR2D " << sigmasq_sampled.t() << endl;
  
  splitsub = in_splits;
  p1 = Xin.n_rows;
  p2 = Xin.n_cols;
  n = yin.n_elem;
  
  y = yin;
  X2d = Xin;
  
  lambda = lambda_in;
  g = gin;
  radius = radiusin;
  
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
    double module_bubbles_radius = radius==-1? -1 : radius*(n_stages-s); 
    bool fix_sigma_module = fix_sigma;
    // fix sigma at last stage
    if(binary_outcome & !fix_sigma){
      if(s == n_stages-1){
        fix_sigma_module = true;
      }
    }
    
    BayesLM2D adding_module(residuals, X2d, 
                            current_split_field, mask_nosplits, lambda, 
                            fix_sigma_module, sigmasq_sampled(s), g, module_bubbles_radius);
    modules.push_back(adding_module);
    //clog << "logmarglik: " << adding_module.logmarglik() << " sigprior: " << sigprior << endl;
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
  }
  
  
  
}

inline void ModularLR2D::change_module(int which_level, const arma::field<arma::mat>& in_splits){
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
    modules[s].change_splits(residuals, current_split_field);
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

inline void ModularLR2D::propose_change_module(int which_level, const arma::field<arma::mat>& in_splits, bool all=false){
  // all=true: keep sigmasq fixed, change centers at all levels. less modular but potential for worse mixing.
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
  
  int limit_here = all ? n_stages : (which_level+1);
  for(unsigned int s = which_level; s<limit_here; s++){
    arma::field<arma::mat> current_split_field(s+1);
    for(unsigned int j=0; j<s+1; j++){
      current_split_field(j) = in_splits(j);
    }
    //clog << "Proposal change: " << endl;
    //clog << "Before: " << modules[s].logmarglik() << endl;
    modules[s].change_splits(residuals, current_split_field, false); // don't resample sigma
    //clog << "After: " << modules[s].logmarglik() << endl;
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

inline void ModularLR2D::confirm_change_module(int which_level, const arma::field<arma::mat>& in_splits, bool all=false){
  splitsub = in_splits;
  
  double sigprior = 0.0;
  if(!fix_sigma){
    for(unsigned int s = 0; s<which_level; s++){
      sigprior += R::dgamma(1.0/modules[s].sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
    }
  }

  arma::vec residuals;
  if(which_level > 0){
    residuals = modules[which_level-1].residuals;
  } else {
    residuals = y;
  }
  
  // update sigma if necessary, confirming the change
  int s = which_level;
  //clog << "Confirm change: " << modules[s].flatmodel.fix_sigma << endl;
  //clog << "Before s l: " <<  modules[s].flatmodel.sigmasq << " " << modules[s].logmarglik() << endl;
  modules[s].chg_y(residuals);
  //clog << "After  s l: " <<  modules[s].flatmodel.sigmasq << " " << modules[s].logmarglik() << endl;
  logliks(s) = modules[s].logmarglik() + sigprior;
  icept_sampled(s) = modules[s].icept_sampled;
  theta_sampled.slice(s) = modules[s].beta_sampled;
  sigmasq_sampled(s) = modules[s].sigmasq_sampled;
  residuals = modules[s].residuals;
  if(!fix_sigma){
    sigprior += R::dgamma(1.0/modules[s].sigmasq_sampled, modules[s].flatmodel.alpha, 1.0/modules[s].flatmodel.beta, 1); //***
  }

  for(unsigned int s = which_level+1; s<n_stages; s++){
    if(all){ // already proposed the split change, just need to update sigmas
      modules[s].chg_y(residuals);
    } else { // also update the splits
      arma::field<arma::mat> current_split_field(s+1);
      for(unsigned int j=0; j<s+1; j++){
        current_split_field(j) = in_splits(j);
      }
      modules[s].change_splits(residuals, current_split_field);
    }
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



class PriorVS{
public:
  arma::vec y;
  arma::mat X;
  int n, p;
  double alpha, beta;
  
  arma::uvec gamma;
  arma::uvec gamma_proposal;
  bmmodels::BayesLM model;
  bmmodels::BayesLM model_proposal;
  double logpost;
  
  arma::vec sampling_order;
  
  void chain(int);
  void change_m(const arma::vec&);
  int mcmc;
  
  arma::vec icept_stored;
  arma::mat gamma_stored;
  arma::mat beta_stored;
  arma::vec sigmasq_stored;
  
  arma::vec priormean;
  
  bool fix_sigma;
  double g;
  double model_prior_par;
  arma::uvec gammaix;
  
  PriorVS(const arma::vec&, const arma::mat&, 
          const arma::vec&, const arma::vec&, 
          double, double, bool, int);
};

inline void PriorVS::chain(int iter){
  mcmc = iter;
  icept_stored = arma::zeros(mcmc);
  beta_stored = arma::zeros(p, mcmc);
  gamma_stored = arma::zeros(p, mcmc);
  sigmasq_stored = arma::zeros(mcmc);
  //clog << mcmc << endl;
  for(int m=0; m<mcmc; m++){
    sampling_order = arma::regspace(0, p-1); // bmrandom::rndpp_shuffle(p_indices);
    for(int j=0; j<p; j++){
      int ix = sampling_order(j);
      gamma_proposal = gamma;
      gamma_proposal(ix) = 1-gamma(ix);
      arma::uvec gammaix_proposal = arma::find(gamma_proposal);
      //clog << arma::size(priormean) << endl;
      //clog << "proposal gamma " << arma::size(gamma_proposal) << " " << arma::size(X.cols(gammaix_proposal)) << endl;
      //clog << arma::size(priormean.elem(gammaix_proposal)) << endl;
      model_proposal = bmmodels::BayesLM(y, X.cols(gammaix_proposal), priormean.elem(gammaix_proposal),
                               alpha, beta, 1.0, fix_sigma, 1.0, g);
      
      //clog << "test  mcmc j " << j << endl;
      double accept_probability = exp(model_proposal.logpost - model.logpost) *
        exp(model_prior_par * (model.p - model_proposal.p));
      accept_probability = accept_probability > 1 ? 1.0 : accept_probability;
      
      int accepted = bmrandom::rndpp_bern(accept_probability);
      if(accepted == 1){
        //clog << "accepted." << endl;
        model = model_proposal;
        gamma = gamma_proposal;
        gammaix = gammaix_proposal;
      }
    }
    
    arma::vec beta_full = arma::zeros(p);
    beta_full.elem(gammaix) = model.b;
    
    icept_stored(m) = model.icept;
    beta_stored.col(m) = beta_full;
    gamma_stored.col(m) = arma::conv_to<arma::vec>::from(gamma);
    sigmasq_stored(m) = model.sigmasq;
  }
}

inline void PriorVS::change_m(const arma::vec& newm){
  model = bmmodels::BayesLM(y, X.cols(gammaix), newm.elem(gammaix),
                  alpha, beta, 1.0, fix_sigma, 1.0, g);
  logpost = model.logpost;
}


inline PriorVS::PriorVS(const arma::vec& yy, 
                        const arma::mat& XX, 
                        const arma::vec& priormeanin,
                        const arma::vec& priorgammain,
                        double gin=-1.0, double model_prior_par_in=1, bool fixsigma=false, int iter=1){
  
  y = yy;
  X = XX;
  mcmc = iter;
  
  p = X.n_cols;
  arma::vec p_indices = arma::linspace<arma::vec>(0, p-1, p);
  n = y.n_elem;
  
  g = gin;
  model_prior_par = model_prior_par_in;
  fix_sigma = fixsigma;
  
  gamma = arma::uvec(p);
  for(int j=0; j<p; j++){
    gamma(j) = 1*bmrandom::boolbern(priorgammain(j));
  }
  priormean = priormeanin; 
  
  gammaix = arma::find(gamma);
  //clog << gammaix << endl;
  //clog << arma::size(X) << " " << arma::size(priormean) << endl;
  model = bmmodels::BayesLM(y, X.cols(gammaix), priormean.elem(gammaix),
                  alpha, beta, 1.0, fix_sigma, 1.0, g);
  logpost = model.logpost;
  //clog << "test  2" << endl;
  chain(mcmc);
  //clog << "test ." << endl;
}


#endif