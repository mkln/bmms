//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef mcmc_helper
#define mcmc_helper

#include <RcppArmadillo.h>
#include <bmtools.h>
//#include "bmmodels2.h"

using namespace std;


class Module {
public:
  
  arma::vec ej;
  arma::vec ej_next;
  arma::mat Xj;
  
  double bmean;
  double icept;
  arma::vec xs;
  
  arma::vec xb_mean, xb_sample;
  //arma::mat Xfull;
  
  arma::mat J_pre;
  arma::mat J_now;
  //arma::mat stretcher;
  //arma::vec normalizer;
  
  arma::vec grid;
  bool fix_sigma;
  
  int n;
  int pj;
  
  arma::mat In;
  arma::mat XtX;
  arma::mat XtXi;
  arma::mat Px;
  arma::mat var_pre;
  arma::mat inv_var_pre;
  arma::vec mean_pre;
  double a;
  double b;
  
  arma::vec Jcs;
  
  double g; // g-prior
  double ridge;
  
  arma::vec msplits;
  
  arma::vec mean_post;
  arma::mat inv_var_post;
  arma::mat var_post;
  
  arma::vec theta_sample;
  
  //arma::vec theta_p_mean;
  arma::vec theta_p_sample;
  
  //arma::mat KsKs;
  //arma::mat KsK;
  
  double beta_post;
  double sigmasq_sample;
  double sigmasq_mean;
  //arma::mat bigsigma;
  
  //arma::mat marglik_module_var; //because there is no previous module
  //arma::mat marglik_integr_var;
  //arma::vec marglik_mean;
  
  void redo(const arma::vec&,
            const arma::mat&,
            const arma::mat&,
            const arma::vec&);
  
  // e, X, Jpre, Jnow, prior mean and var, previously sampled theta
  Module(arma::mat&, arma::vec&, arma::mat&, 
         double, arma::mat&, arma::mat&, arma::vec&, arma::vec&, arma::vec&, arma::vec&, 
         double, double, double);
  // empty constructor
  Module();
};

class ModularLinReg {
public:
  
  double rad;
  arma::mat In;
  // input data
  int n;
  int p;
  double g_prior;
  arma::vec y;
  arma::mat X;
  double intercept;
  bool fix_sigma;
  bool nested;
  double fixed_sigma_v;
  
  double structpar;
  
  arma::vec xb;
  
  // limit to number of possible stages
  int max_stages;
  
  // grid splits
  int n_splits;
  int n_stages;
  arma::field<arma::vec> split_seq;
  arma::field<arma::vec> bigsplit;
  arma::vec grid;
  arma::vec pones;
  arma::vec cumsplit;
  
  // resolution structure
  arma::field<arma::mat> X_field;
  arma::field<arma::mat> J_field;
  //arma::field<arma::vec> m_field;
  //arma::field<arma::mat> M_field;
  
  double a, b;
  
  std::vector<Module> modules;
  
  //arma::field<arma::mat> mod_reshaper;
  arma::mat faux_J_previous;
  arma::vec faux_theta_previous_sample;
  
  bool fixed_splits;
  // operations on modules
  // new module only needs new splits
  void add_new_module(arma::vec&);
  void delete_last_module();
  void change_module(int, arma::vec&);
  void redo();
  void change_all(arma::field<arma::vec>&);
  
  // information on current setup
  arma::vec loglik;
  
  // mu_field(s) is the BMMS point estimator for the linea regression coeff using grid J_field(s)
  arma::field<arma::vec> mu_field; // list of posterior cumulative means of theta, the multiscale parameter
  arma::field<arma::vec> the_sample_field; // list of cumulative samples (dim p) of theta, for different scales
  arma::field<arma::vec> theta_p_scales; // list of samples (dim p) of theta
  
  //arma::field<arma::mat> bigsigma;
  //arma::field<arma::mat> bigsigma_of_incr;
  arma::vec sigmasq_scales;
  
  int opt;
  // constructor
  // y, X, list of splits for each stage, limit to stages
  ModularLinReg(const arma::vec&, const arma::mat&, 
                double, 
                const arma::field<arma::vec>&, double, int, double, bool,
                double, double, double);
  
};

class ModularVS {
public:
  int K;
  int mcmc;
  
  //std::vector<VSModule> varsel_modules;
  
  arma::vec y;
  int n;
  arma::field<arma::mat> Xall;
  arma::vec resid;
  
  //ModularVS(const arma::vec&, const arma::field<arma::mat>&, int, double, arma::vec);
  ModularVS(const arma::vec&, const arma::field<arma::mat>&, 
            const arma::field<arma::vec>&,
            int, arma::vec, arma::vec, bool);
  
  arma::mat intercept;
  arma::field<arma::mat> beta_store;
  arma::field<arma::mat> gamma_store;
  arma::field<arma::vec> gamma_start;
};

class VarSelOps{
public:
  arma::vec y;
  arma::mat X;
  int n, p;
  
  arma::uvec gamma;
  arma::uvec gamma_proposal;
  bmmodels::BayesLMg sampled_model;
  double loglik;
  
  //arma::vec gamma_start_prior; //prior prob of values to start from
  
  arma::vec sampling_order;
  double model_prior_par;
  bool fixsigma;
  double g;
  int mcmc;
  
  
  void forward(int);
  void change_y(const arma::vec&);
  
  arma::uvec gammaix;
  
  arma::vec icept_stored;
  arma::mat gamma_stored;
  arma::mat beta_stored;
  arma::vec sigmasq_stored;
  
  
  arma::vec linear_predictor;
  
  VarSelOps(const arma::vec&, const arma::mat&, const arma::vec&, double, double, bool, int);
};

class ModularVS2 {
public:
  int K;
  int mcmc;
  
  //std::vector<VSModule> varsel_modules;
  
  arma::vec y;
  int n;
  arma::field<arma::mat> Xall;
  arma::vec resid;
  arma::mat z_all;
  arma::mat z_all_proposed;
  
  //ModularVS(const arma::vec&, const arma::field<arma::mat>&, int, double, arma::vec);
  ModularVS2(const arma::vec&, const arma::field<arma::mat>&, 
            const arma::field<arma::vec>&,
            int, arma::vec, arma::vec, bool);
  
  std::vector<VarSelOps> modules;
  std::vector<VarSelOps> proposed_modules;
  
  arma::vec logliks;
  arma::mat logliks_stored;
  arma::vec proposed_logliks;
    
  arma::mat intercept;
  arma::field<arma::mat> beta_store;
  arma::field<arma::mat> gamma_store;
  arma::field<arma::vec> gamma_start;
};


double log_mvn_density(arma::vec x, arma::vec mean, arma::mat covar);

double modular_loglik1(arma::vec& y, arma::vec& marglik_mean, arma::mat& varloglik, arma::vec& sigmasq_scales, int n_stages);
double modular_loglik2(arma::vec& y, arma::mat& mean_post, arma::mat& inv_var_post, double a, double b);
double modular_loglik0(arma::vec& y, double a, double b);

inline double bdet(const arma::mat& X){
  double val, sign;
  arma::log_det(val, sign, X);
  return val;
}

double modular_loglikn(const arma::vec& x, const arma::mat& Si);
double totsplit_prior2_ratio(int tot_split_prop, int tot_split_orig, int norp, int ss, double lambda_prop);
double split_struct_ratio2(const arma::field<arma::vec>& proposed, const arma::field<arma::vec>& original,
                                  int stage, int p, double param);

double totsplit_prior_ratio(int tot_split_prop, int tot_split_orig, int norp, int ss, double lambda_prop);

double splitpar_prior(double x, int tot_split, int norp, int ss);

double totstage_prior_ratio(int tot_stage_prop, int tot_stage_orig, int norp, int curr_n_splits, int direction);


// log density of mvnormal mean 0 -- only useful in ratios with gpriors
inline double m0mvnorm_dens_gratio(double yy, double yPxy, double g, double p){
  return -0.5*p*log(g+1.0) + 0.5*g/(g+1.0) * yPxy - 0.5*yy;
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
//-- only useful in ratios with gpriors
// and gprior for b
inline double clm_marglik_gratio(double yy, double yPxy, double g, int n, double p, double a, double b){
  return -0.5*p*log(g+1.0) - (a+n/2.0)*log(b + 0.5*(yy - g/(g+1.0) * yPxy));
}


#endif

