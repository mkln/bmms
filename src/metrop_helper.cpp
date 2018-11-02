//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#include "metrop_helper.h"
#include <RcppArmadilloExtensions/sample.h>

using namespace std;

int rndpp_sample1_comp(arma::vec x, int p, int current_split, double decay=4.0){
  /* vector x = current splits, p = how many in total
  * this returns 1 split out of the complement 
  * decay controls how far the proposed jump is going to be from the current
  * decay=1 corresponds to uniform prob on all availables
  * if current_split=-1 then pick uniformly
  */
  //double decay = 5.0;
  arma::vec all = arma::linspace(0, p-1, p);
  arma::vec avail = bmms_setdiff(all, x);
  //cout << avail << endl;
  arma::vec probweights;
  if(current_split == -1){
    probweights = arma::ones(arma::size(avail));
  } else {
    probweights = arma::exp(arma::abs(avail - current_split) * log(1.0/decay));
  }
  if(avail.n_elem > 0){
    arma::vec out = Rcpp::RcppArmadillo::sample(avail, 1, true, probweights); 
    return out(0);
  } else {
    return -1;
  }
}

arma::vec rndpp_shuffle(const arma::vec& x){
  /* vector x = a vector
  output = reshuffled vector
  */
  //double decay = 5.0;
  return Rcpp::RcppArmadillo::sample(x, x.n_elem, false); 
}


arma::uvec sample_index(const int& n, const int &vsize){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, vsize-1, vsize);
  arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, n, false);
  return out;
}

int sample_one_int(const int &vsize){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, vsize-1, vsize);
  int out = (Rcpp::RcppArmadillo::sample(sequence, 1, false))(0);
  return out;
}


arma::mat reshaper(arma::field<arma::mat> J_field, int s){
  arma::mat stretcher = (J_field(s).t() * J_field(s-1));
  arma::mat normalizer = col_sums(J_field(s));
  //stretcher.transform( [](double val) { return (val>0 ? 1 : 0); } );
  for(unsigned int j=0; j<stretcher.n_rows; j++){
    stretcher.row(j) = stretcher.row(j)/normalizer(j);
  }
  return(stretcher);
}

double log_mvn_density(arma::vec x, arma::vec mean, arma::mat covar){
  //cout << arma::size(x) << " " << arma::size(mean) << " " << arma::size(covar) << endl;
  double val, sign, cov_logdet;
  arma::log_det(val, sign, covar);
  cov_logdet = log(sign) + val;
  return( -(.5*x.n_elem)*log(2*M_PI) -.5*cov_logdet -.5*arma::conv_to<double>::from( (x-mean).t()*arma::inv_sympd(covar)*(x-mean) ) );
}


double modular_loglik1(arma::vec& y, arma::vec& marglik_mean, arma::mat& varloglik, arma::vec& sigmasq_scales, int n_stages){
  double result = log_mvn_density(y, marglik_mean, varloglik) - arma::accu(log(sigmasq_scales.subvec(0, n_stages-1)));
  return result;
}

double modular_loglik2(arma::vec& y, arma::mat& mean_post, arma::mat& inv_var_post, double a, double b){
  int p1 = inv_var_post.n_cols;
  int n = y.n_elem;
  double result = (n-p1)*log(n*1.0) - (a+n/2.0) * log(b + .5*arma::conv_to<double>::from(y.t()*y - mean_post.t() * inv_var_post * mean_post));
  return result;
}

double modular_loglik0(arma::vec& y, double a, double b){
  int n = y.n_elem;
  double result = (n)*log(n*1.0) - (a+n/2.0) * log(b + .5*arma::conv_to<double>::from(y.t()*y));
  return result;
}


double modular_loglikn(const arma::vec& x, const arma::mat& Si){
  int p = Si.n_cols;
  double normcore =  arma::conv_to<double>::from(x.t() * Si * x);
  double normconst = - p/2.0 * log(2*M_PI) + .5 * log(arma::det(Si));
  return normconst - 0.5 * (normcore);
}

Module::Module(){
  a = 0;
  b = 0;
  n = 0;
  pj = 0;
  g = 0;
  ridge = 0;
  ej = arma::zeros(0);
  Xj = arma::zeros(0,0);
  Xfull = arma::zeros(0,0);
  J_pre = arma::zeros(0,0);
  J_now = arma::zeros(0,0);
  mean_pre = arma::zeros(0);
  var_pre = arma::zeros(0,0);
  inv_var_pre = arma::zeros(0,0);
  stretcher = arma::zeros(0,0);
  inv_var_post = arma::zeros(0,0);
  var_post = arma::zeros(0,0);
  mean_post = arma::zeros(0);
  beta_post = 0.0;
  sigmasq_sample = 0.0;
  theta_sample = arma::zeros(0);
  
  //theta_p_mean = arma::zeros(0);
  theta_p_sample = arma::zeros(0);
  ej_next = arma::zeros(0);
  marglik_module_var = arma::zeros(0,0);
  marglik_integr_var = arma::zeros(0,0);
  marglik_mean = arma::zeros(0);
  //bigsigma = arma::zeros(0,0);
  Px = arma::zeros(0,0);
  //KsKs = arma::zeros(0,0);
  //KsK = arma::zeros(0,0);
}


Module::Module(arma::mat& X_full, arma::vec& e, arma::mat& X, double g_prior,
               arma::mat& Jpre, arma::mat& Jnow, arma::vec& mean_post_pre, arma::vec& theta_sample_pre,
               arma::vec& ingrid, arma::vec& splits, int kernel_type, bool fixed_sigma=false,
               double ain=2.1, double bin=1.1){
  //cout << "ModularLinReg :: module 0 " << endl;
  Xfull = X_full;
  ej = e;
  Xj = X;
  n = e.n_elem;
  pj = Xj.n_cols;
  
  fix_sigma = fixed_sigma;
  g = g_prior;
  
  msplits = splits;
  ktype = kernel_type;
  
  J_now = Jnow;
  
  ridge = 1.0;
  
  XtX = Xj.t() * Xj;
  //clog << J_now << endl;
  //clog << XtX << endl;
  XtXi = arma::inv_sympd(XtX + (ridge*arma::eye(Xj.n_cols, Xj.n_cols))); //(ridge*arma::eye(Xj.n_cols, Xj.n_cols))); //(0.1*(pj>0.01*n)*arma::eye(Xj.n_cols, Xj.n_cols)));
  
  Px = Xj * XtXi * Xj.t();
  
  //clog << "see me?" << endl;
  mean_pre = arma::zeros(Xj.n_cols);
  var_pre = g*XtXi;
  var_post = g/(g+1.0) * (XtXi);
  inv_var_post = (g+1.0)/n * XtX;
  mean_post = var_post * Xj.t() * ej;
  In = arma::eye(n,n);
  a = ain; // parametrization: a = mean^2 / variance + 2
  b = bin;  //                  b = (mean^3 + mean*variance) / variance
  
  //clog << a << " " << b << endl;
  if(!fix_sigma){
    beta_post = arma::conv_to<double>::from(.5*(
      mean_pre.t() * (1.0/g * XtX) * mean_pre + 
      ej.t()*ej - 
      mean_post.t() * ( (g+1.0)/g * XtX) * mean_post)
    );
    
    //clog << a + n/2.0 << " " << (b + beta_post) << endl;
    sigmasq_sample = 1.0/rndpp_gamma(a + n/2.0, 1.0/(b + beta_post));
    sigmasq_mean = (b+beta_post)/(a+n/2.0+1);
  } else {
    sigmasq_sample = 1.0;
  }
  
  theta_sample = rndpp_mvnormal(1, mean_post, sigmasq_sample*var_post).t();
  
  //clog << sigmasq_sample << " " << sigmasq_mean << endl;
  //module variance HERE to be used with SUBSEQUENT models
  // change to sigmasq_sample for MCMC
  //marglik_module_var = sigmasq_sample * (Xj * var_pre * Xj.t()); //sigmasq_sample * (Xj * var_post * Xj.t()); 
  
  //integrated ml variance to be used with previous module_vars
  //marglik_integr_var = sigmasq_sample * (In + Xj * var_pre * Xj.t());
  marglik_mean = Xfull * (Jpre * mean_post_pre + Jnow * mean_pre);
  //
  
  //theta_p_mean = (Jpre * mean_post_pre + Jnow * mean_post);
  theta_p_sample = (Jpre * theta_sample_pre + Jnow * theta_sample);
  
  grid = ingrid;
  
  //if(MCMCSWITCH == 1){
  marglik_module_var = sigmasq_sample * (Xj * var_post * Xj.t()); 
  //integrated ml variance to be used with previous module_vars
  marglik_integr_var = sigmasq_sample * (In + Xj * var_pre * Xj.t());
  //}
  
  //bigsigma = sigmasq_sample * var_post;
  
  ej_next = ej - Xj * theta_sample;
}

void Module::redo(arma::vec& e){
  ej = e;
  
  cout << "redoing module" << endl;
  
  if(!fix_sigma){
    beta_post = arma::conv_to<double>::from(.5*(
      ej.t() * (In - g/(g+1.0) * Px) * ej 
    ));
    
    cout << "beta post " << beta_post << endl;
    
    
    sigmasq_sample = 1.0/rndpp_gamma(a + n/2.0, 1.0/(b + beta_post)); //1.0/rndpp_gamma(n/2.0, 1.0/beta_post);
  } else {
    sigmasq_sample = 1.0;
  }
  
  mean_post = var_post * Xj.t() * ej;
  cout << "sigmasq sample " << sigmasq_sample << endl;
  theta_sample = rndpp_mvnormal(1, mean_post, sigmasq_sample*var_post).t();
  
  //if(MCMCSWITCH == 1){
  marglik_module_var = sigmasq_sample * (Xj * var_post * Xj.t()); 
  //integrated ml variance to be used with previous module_vars
  marglik_integr_var = sigmasq_sample * (In + Xj * var_pre * Xj.t());
  
  //}
  
  //bigsigma = sigmasq_sample * var_post;
  
  ej_next = ej - Xj * theta_sample;
  
  cout << "redone module " << endl;
}


ModularLinReg::ModularLinReg(const arma::vec& yin, 
                             const arma::mat& Xin, 
                             double g, 
                             const arma::field<arma::vec>& in_splits, 
                             int kernel, int set_max_stages, 
                             bool fixed_sigma=false, bool fixed_grids = true,
                             double ain=2.1, double bin=1.1){
  
  kernel_type = kernel;
  // kernel 0 = step functions; 1 = gaussian
  
  a = ain;
  b = bin;
  
  fixed_splits = fixed_grids;
  fix_sigma = fixed_sigma;
  intercept = arma::mean(yin);
  
  y = yin-intercept;
  X = Xin;
  
  max_stages = set_max_stages;
  //cout << "in splits: " << endl << in_splits << endl;
  
  loglik = arma::zeros(max_stages);
  n = y.n_elem;
  p = X.n_cols;
  In = arma::eye(n, n);
  g_prior = g;
  
  split_seq = in_splits;
  n_stages = 1;
  
  bigsplit = arma::field<arma::vec>(max_stages);
  //arma::vec bigsplit_sizes = arma::zeros(split_seq.n_elem);
  cumsplit = arma::zeros(max_stages);
  
  bigsplit(0) = split_seq(0);
  cumsplit(0) = bigsplit(0).n_elem;
  for(unsigned int s=1; (s<split_seq.n_elem); s++){
    if(split_seq(s).n_elem>0){
      n_stages ++;
      bigsplit(s) = arma::join_vert(bigsplit(s-1), split_seq(s));
      cumsplit(s) = bigsplit(s).n_elem;
    } else {
      break;
    }
  }
  // initialization of structural variables
  J_field = arma::field<arma::mat>(max_stages); //n_stages+1
  X_field = arma::field<arma::mat>(max_stages); //J_field.n_elem
  //mod_reshaper = arma::field<arma::mat>(max_stages-1);
  
  // initialization of prior variables
  m_field = arma::field<arma::vec>(max_stages);
  M_field = arma::field<arma::mat>(max_stages);
  mu_field = arma::field<arma::vec>(max_stages);
  the_sample_field = arma::field<arma::vec>(max_stages);
  
  // structure of module 0
  grid = arma::linspace(0, p-1, p);
  pones = arma::ones(p, 1);
  
  J_field(0) = multi_split(pones, split_seq(0), p);
  
  X_field(0) = X * J_field(0);
  //
  theta_p_scales = arma::field<arma::vec>(max_stages);
  sigmasq_scales = arma::zeros(max_stages);
  //bigsigma = arma::field<arma::mat>(max_stages);
  //bigsigma_of_incr = arma::field<arma::mat>(max_stages);
  
  // structure of all other modules
  //cout << "ModularLinReg :: creating other elements after 0 " << endl;
  for(unsigned int j=1; j<n_stages; j++){
    J_field(j) = multi_split(pones, bigsplit(j), p);
    X_field(j) = X * J_field(j);
  }
  
  // e, X, Jpre, Jnow, prior mean and var, previously sampled theta
  faux_J_previous = arma::zeros(J_field(0).n_rows);
  faux_theta_previous_sample = arma::zeros(1);
  
  cout << "-adding first module" << endl;
  Module adding_module(X, y, X_field(0), g_prior, faux_J_previous, J_field(0), 
                       faux_theta_previous_sample, faux_theta_previous_sample, grid, 
                       bigsplit(0), kernel_type, false, a, b);
  cout << "added." << endl;
  modules.push_back(adding_module);
  
  mu_field(0) = modules[0].mean_post; //modules[0].J_now * modules[0].mean_post;
  the_sample_field(0) = modules[0].J_now * modules[0].theta_sample;
  
  theta_p_scales(0) = modules[0].theta_p_sample;
  sigmasq_scales(0) = modules[0].sigmasq_sample;
  //bigsigma(0) = modules[0].bigsigma;
  //bigsigma_of_incr(0) = modules[0].bigsigma;
  
  
  for(unsigned int s=1; s<n_stages; s++){
    //cout << "-adding other module" << endl;
    adding_module = Module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), J_field(s-1), J_field(s), 
                           modules[s-1].mean_post, modules[s-1].theta_sample,
                           grid, bigsplit(s), kernel_type, false, a, b);
    //cout << "--added." << endl;
    modules.push_back(adding_module);
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    
    //mod_reshaper(s-1) = reshaper(J_field, s);
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
  }
  
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglikn(modules[s].ej, In - modules[s].Xj*modules[s].inv_var_post*modules[s].Xj.t());
        
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){
        //clog << s << endl;
        //clog << modules[s].ej << " " << endl <<  modules[s].mean_post << endl <<
        //  arma::det(modules[s].inv_var_post) << endl << modules[s].a << " " << modules[s].b << endl;
            
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
        
      }
    }
  } 
  cout << "ModularLinReg created " << endl;
}

void ModularLinReg::add_new_module(arma::vec& new_splits){
  //cout << "new Module for ModularLinReg" << endl;
  n_stages++;
  int s = n_stages-1;
  
  split_seq(s) = new_splits;
  bigsplit(s) = arma::join_vert(bigsplit(s-1), new_splits);
  cumsplit(s) = bigsplit(s).n_elem;
  
  J_field(s) = multi_split(pones, bigsplit(s), p);
  
  X_field(s) = X * J_field(s);
  
  Module adding_module = Module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), J_field(s-1), J_field(s), 
                                modules[s-1].mean_post, modules[s-1].theta_sample,
                                grid, bigsplit(s), kernel_type, false, a, b);
  modules.push_back(adding_module);
  
  theta_p_scales(s) = modules[s].theta_p_sample;
  mu_field(s) = modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
  the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
  sigmasq_scales(s) = modules[s].sigmasq_sample;
  
  //mod_reshaper(s-1) = reshaper(J_field, s);
  //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
  //  modules[s].bigsigma);
  //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
  //  modules[s].bigsigma);
  
  /*
   arma::mat varloglik = modules[n_stages-1].marglik_integr_var;
   for(int s=0; s<n_stages-1; s++){
   //if(0){ //n_stages>1){
   varloglik += modules[s].marglik_module_var;
   }
   //}
   loglik = modular_loglik(y, modules[n_stages-1].marglik_mean, varloglik, sigmasq_scales, n_stages);
   */
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglikn(modules[s].ej, In - modules[s].Xj*modules[s].inv_var_post*modules[s].Xj.t());
        
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
      }
    }
  } 
  
}


void ModularLinReg::delete_last_module(){
  n_stages -= 1;
  
  bigsplit(n_stages) = arma::zeros(0);
  split_seq(n_stages) = arma::zeros(0);
  cumsplit(n_stages) = 0;
  J_field(n_stages) = arma::zeros(0,0);
  X_field(n_stages) = arma::zeros(0,0);
  m_field(n_stages) = arma::zeros(0);
  M_field(n_stages) = arma::zeros(0,0);
  
  modules.pop_back();
  
  /*
  arma::mat varloglik = modules[n_stages-1].marglik_integr_var;
  for(int s=0; s<n_stages-1; s++){
  //if(0){ //n_stages>1){
  varloglik += modules[s].marglik_module_var;
  }
  //}
  loglik = modular_loglik(y, modules[n_stages-1].marglik_mean, varloglik, sigmasq_scales, n_stages);
  */
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglikn(modules[s].ej, In - modules[s].Xj*modules[s].inv_var_post*modules[s].Xj.t());
        
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
      }
    }
  } 
}


void ModularLinReg::redo(){
  // resample from all modules 
  
  //for(int j=0; j<10000; j++){
  modules[0].redo(y);
  
  mu_field(0) = modules[0].mean_post; //modules[0].J_now * modules[0].mean_post;
  the_sample_field(0) = modules[0].J_now * modules[0].theta_sample;
  
  theta_p_scales(0) = modules[0].theta_p_sample;
  sigmasq_scales(0) = modules[0].sigmasq_sample;
  //bigsigma(0) = modules[0].bigsigma;
  //bigsigma_of_incr(0) = modules[0].bigsigma;
  
  
  for(unsigned int s=1; s<n_stages; s++){
    cout << "-adding other module" << endl;
    modules[s].redo(modules[s-1].ej_next);
    cout << "--added." << endl;
    
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = modules[s].mean_post;//modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    
    //if(!fixed_splits){
    //  mod_reshaper(s-1) = reshaper(J_field, s);
    //}
    
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    
  }
  
  /*
   arma::mat varloglik = modules[n_stages-1].marglik_integr_var;
   for(int s=0; s<n_stages-1; s++){
   //if(0){ //n_stages>1){
   varloglik += modules[s].marglik_module_var;
   }
   //}
   loglik = modular_loglik(y, modules[n_stages-1].marglik_mean, varloglik, sigmasq_scales, n_stages);
   */
  //clog << "getting here?" << endl;
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglikn(modules[s].ej, In - modules[s].Xj*modules[s].inv_var_post*modules[s].Xj.t());
        
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
      }
    }
  } 
}


void ModularLinReg::change_all(arma::field<arma::vec>& new_splitseq){
  split_seq = new_splitseq;
  n_stages = 1;
  
  bigsplit = arma::field<arma::vec>(max_stages);
  //arma::vec bigsplit_sizes = arma::zeros(split_seq.n_elem);
  cumsplit = arma::zeros(max_stages);
  
  bigsplit(0) = split_seq(0);
  cumsplit(0) = bigsplit(0).n_elem;
  for(unsigned int s=1; (s<split_seq.n_elem); s++){
    if(split_seq(s).n_elem>0){
      n_stages ++;
      bigsplit(s) = arma::join_vert(bigsplit(s-1), split_seq(s));
      cumsplit(s) = bigsplit(s).n_elem;
    } else {
      break;
    }
  }
  // initialization of structural variables
  J_field = arma::field<arma::mat>(max_stages); //n_stages+1
  X_field = arma::field<arma::mat>(max_stages); //J_field.n_elem
  //mod_reshaper = arma::field<arma::mat>(max_stages-1);
  
  // initialization of prior variables
  m_field = arma::field<arma::vec>(max_stages);
  M_field = arma::field<arma::mat>(max_stages);
  mu_field = arma::field<arma::vec>(max_stages);
  the_sample_field = arma::field<arma::vec>(max_stages);
  
  // structure of module 0
  grid = arma::linspace(0, p-1, p);
  pones = arma::ones(p, 1);
  
  J_field(0) = multi_split(pones, split_seq(0), p);
  
  X_field(0) = X * J_field(0);
  //
  theta_p_scales = arma::field<arma::vec>(max_stages);
  sigmasq_scales = arma::zeros(max_stages);
  //bigsigma = arma::field<arma::mat>(max_stages);
  //bigsigma_of_incr = arma::field<arma::mat>(max_stages);
  
  // structure of all other modules
  //cout << "ModularLinReg :: creating other elements after 0 " << endl;
  for(unsigned int j=1; j<n_stages; j++){
    J_field(j) = multi_split(pones, bigsplit(j), p);
    X_field(j) = X * J_field(j);
  }
  
  // e, X, Jpre, Jnow, prior mean and var, previously sampled theta
  faux_J_previous = arma::zeros(J_field(0).n_rows);
  faux_theta_previous_sample = arma::zeros(1);
  
  cout << "-adding first module" << endl;
  Module adding_module(X, y, X_field(0), g_prior, faux_J_previous, J_field(0), 
                       faux_theta_previous_sample, faux_theta_previous_sample, grid, bigsplit(0), kernel_type, false, a, b);
  cout << "added." << endl;
  modules.push_back(adding_module);
  
  mu_field(0) = modules[0].mean_post; //modules[0].J_now * modules[0].mean_post;
  the_sample_field(0) = modules[0].J_now * modules[0].theta_sample;
  
  theta_p_scales(0) = modules[0].theta_p_sample;
  sigmasq_scales(0) = modules[0].sigmasq_sample;
  //bigsigma(0) = modules[0].bigsigma;
  //bigsigma_of_incr(0) = modules[0].bigsigma;
  
  for(unsigned int s=1; s<n_stages; s++){
    //cout << "-adding other module" << endl;
    adding_module = Module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), J_field(s-1), J_field(s), 
                           modules[s-1].mean_post, modules[s-1].theta_sample,
                           grid, bigsplit(s), kernel_type, false, a, b);
    //cout << "--added." << endl;
    modules.push_back(adding_module);
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    
    //mod_reshaper(s-1) = reshaper(J_field, s);
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
  }
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglikn(modules[s].ej, In - modules[s].Xj*modules[s].inv_var_post*modules[s].Xj.t());
        
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
      }
    }
  } 
  cout << "ModularLinReg created " << endl;
}



void ModularLinReg::change_module(int whichone, arma::vec& new_splits){
  // the idea is to change the structure of the selected model, 
  // and resample from all new subsequent modules 
  
  //cout << "changing the splits for module-stage " << whichone <<endl;
  split_seq(whichone) = new_splits;
  
  for(unsigned int s=whichone; (s<n_stages); s++){ 
    //starting from whichone, but n_stages is the same here
    bigsplit(s) = arma::join_vert(bigsplit(s-1), split_seq(s));
    cumsplit(s) = bigsplit(s).n_elem;
    J_field(s) = multi_split(pones, bigsplit(s), p);
    
    X_field(s) = X * J_field(s);
    
    Module adding_module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), J_field(s-1), J_field(s), 
                         modules[s-1].mean_post, modules[s-1].theta_sample, grid, bigsplit(s), kernel_type, false, a, b);
    modules[s] = adding_module;
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    
    
    //mod_reshaper(s-1) = reshaper(J_field, s);
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    
  }
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglikn(modules[s].ej, In - modules[s].Xj*modules[s].inv_var_post*modules[s].Xj.t());
        
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){ 
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
      }
    }
  } 
}
double totsplit_prior2_ratio(int tot_split_prop, int tot_split_orig, int norp, int ss, double lambda_prop){
  double ck = lambda_prop * pow(2.0,ss);
  double proposal = (- ck * tot_split_prop * std::log(tot_split_prop));
  double orig = (- ck * tot_split_orig * std::log(tot_split_orig));
  return exp(proposal - orig);
}

double totsplit_prior_ratio(int tot_split_prop, int tot_split_orig, int norp, int ss, double lambda_prop){
  //lambda_prop is 1/variance;
  double means = pow(2, ss);
  return exp(-lambda_prop/2.0 * pow(tot_split_prop - means, 2) + lambda_prop/2.0 * pow(tot_split_orig - means, 2) ); //prior_ratio;
}

double splitpar_prior(double x, int tot_split, int norp, int ss){
  //lambda_prop is 1/variance;
  double means = pow(2, ss);
  return -x/2.0 * pow(tot_split - means, 2); //prior_ratio;
  //return exp(-0.0-prior_levs);
}

double totstage_prior_ratio(int tot_stage_prop, int tot_stage_orig, int norp, int curr_n_splits, int direction){
  return 1.0;
}

arma::field<arma::vec> splits_truncate(arma::field<arma::vec> splits, int k){
  int k_effective = splits.n_elem > k ? k : splits.n_elem;
  if(k_effective<1){
    k_effective=1;
  }
  arma::field<arma::vec> splits_new(k_effective);
  for(int i=0; i<k_effective; i++){
    splits_new(i) = splits(i);
  }
  return splits_new;
}