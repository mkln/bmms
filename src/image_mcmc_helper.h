//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef image_voronoi
#define image_voronoi

#include <RcppArmadillo.h>
#include "metrop_helper.h"

double blm_marglik(arma::vec& y, arma::mat& mean_post, arma::mat& inv_var_post, double a, double b);

arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m);

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

arma::field<arma::mat> load_splits(int maxlevs);

arma::field<arma::mat> merge_splits(arma::field<arma::mat>& old_splits, arma::field<arma::mat> new_splits);
double gammaprior_mhr(double new_val, double old_val, double alpha=100, double beta=.5);

double struct2d_prior_ratio(const arma::field<arma::vec>& proposed, const arma::field<arma::vec>& original,
                            int stage, int p, double param);

#endif