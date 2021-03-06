//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]


#include "metrop_helper.h"

using namespace std;



ModularVS::ModularVS(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, 
                     const arma::field<arma::vec>& starting,
                     int mcmc_in,
                     arma::vec gg, 
                     arma::vec module_prior_par, bool binary=false){
  
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
      gamma_start(j)(h) = bmrandom::rndpp_bern(0.1);
    }
    beta_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
    gamma_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
  }
  
  for(int m=0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
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
      
      bmmodels::VarSelMCMC onemodule(resid, Xall(j), gamma_start(j), gg(j), module_prior_par(j), binary?true:false, 1);
      
      //varsel_modules.push_back(onemodule);
      intercept(j, m) = onemodule.icept_stored(0);// onemodule.intercept;
      xb_cumul = xb_cumul + Xall(j) * onemodule.beta_stored.col(0) + onemodule.icept_stored(0);
      resid = (binary?z:y) - xb_cumul;
      //clog << "  beta store" << endl;
      beta_store(j).col(m) = onemodule.beta_stored.col(0);
      //clog << "  gamma store" << endl;
      gamma_store(j).col(m) = onemodule.gamma_stored.col(0);
      //clog << "  gamma start" << endl;
      gamma_start(j) = onemodule.gamma_stored.col(0);
      //clog << gamma_start(1) << endl;
    }
    
    if(binary){
      //z = bmrandom::mvtruncnormal_eye1(xb_cumul, 
      //                                 trunc_lowerlim, trunc_upperlim).col(0);
      arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), xb_cumul);
      z = 1.0/w % (y-.5);
    }
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        clog << m << " " << max(abs(z)) << endl;
      } 
    }
  }
}

ModularVS2::ModularVS2(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, 
                       const arma::field<arma::vec>& starting,
                       int mcmc_in,
                       arma::vec gg, 
                       arma::vec module_prior_par, bool binary=false){
  
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
  logliks = arma::zeros(K);
  proposed_logliks = arma::zeros(K);
  
  logliks_stored = arma::zeros(K, mcmc);
  
  z_all = arma::zeros(n, K);
  
  intercept = arma::zeros(K, mcmc);
  beta_store = arma::field<arma::mat>(K);
  gamma_store = arma::field<arma::mat>(K);
  gamma_start = starting;
  
  for(int j=0; j<K; j++){
    gamma_start(j) = arma::zeros(Xall(j).n_cols);
    for(unsigned int h=0; h<Xall(j).n_cols; h++){
      gamma_start(j)(h) = bmrandom::rndpp_bern(0.1);
    }
    beta_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
    gamma_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
  }
  
  clog << "init";
  
  for(int j=0; j<K; j++){
    if(j == 0){
      if(binary){
        arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), arma::zeros(y.n_elem));
        z = 1.0/w % (y-.5);
        z_all.col(j) = z;
      } else {
        z_all.col(j) = y;
      }
    } else {
      z_all.col(j) = z_all.col(j-1) - modules[j-1].linear_predictor;
    }
    VarSelOps adding_module(z_all.col(j), Xall(j), gamma_start(j), gg(j), module_prior_par(j), binary?true:false, 0); // no mcmc
    modules.push_back(adding_module);
    proposed_modules.push_back(adding_module);
    logliks(j) = modules[j].loglik;
    proposed_logliks(j) = proposed_modules[j].loglik;
  }
  clog << "." << endl;
  int accepted = 0;
  
  for(int m=0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    //clog << "> m: " << m << endl;
    z_all_proposed = z_all;
    proposed_modules = modules;
    
    for(int j=0; j<K; j++){
      //clog << "j: " << j << endl;
      // create alternative
      //proposed_modules = modules;
      
      for(int cc=0; cc<Xall(j).n_cols; cc++){
        // update each variable individually
        // sample gammas from alternative
        proposed_modules[j] = modules[j];
        proposed_modules[j].forward(cc);
        // get logpost from following moduels
        proposed_logliks(j) = proposed_modules[j].loglik; 
        
        for(int s=j+1; s<K; s++){
          // assign new residuals on j+1:K and get logposts
          //clog << "s: " << s << endl;
          //proposed_modules[s] = modules[s];
          z_all_proposed.col(s) = z_all_proposed.col(s-1) - proposed_modules[s-1].linear_predictor;
          proposed_modules[s].change_y(z_all_proposed.col(s));
          proposed_logliks(s) = proposed_modules[s].loglik;
          if((j==0) & (cc<2)){
            //clog << proposed_modules[s].loglik << endl;
          }
        }
        
        double prob = exp(arma::accu(proposed_logliks - logliks));
        //if(cc < 2){
        //  clog << "--" << endl;
        //  clog << "j: " << j << " " << proposed_logliks.t() << " " << logliks.t() << endl;
        //}
        
        prob = prob > 1 ? 1 : prob;
        int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
        if(accepted_proposal == 1){
          // accept all changes to j:K modules
          modules = proposed_modules;
          logliks = proposed_logliks;
          z_all = z_all_proposed;
          accepted++;
        } else {
          // reset changes
          proposed_modules = modules;
          proposed_logliks = logliks;
          z_all_proposed = z_all;
        }
      }
      // storage
      //clog << "  beta store" << endl;
      beta_store(j).col(m) = modules[j].beta_stored.col(0);
      //clog << "  gamma store" << endl;
      gamma_store(j).col(m) = modules[j].gamma_stored.col(0);
      //clog << "  gamma start" << endl;
      gamma_start(j) = modules[j].gamma_stored.col(0);
      intercept(j, m) = modules[j].icept_stored(0);// onemodule.intercept;
      logliks_stored(j, m) = logliks(j);
    }
    
    for(int j=0; j<K; j++){
      if(j == 0){
        if(binary){
          arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), modules[j].linear_predictor);
          z = 1.0/w % (y-.5);
          z_all.col(j) = z;
        } else {
          z_all.col(j) = y;
        }
      } else {
        z_all.col(j) = z_all.col(j-1) - modules[j-1].linear_predictor;
      }
      modules[j] = VarSelOps(z_all.col(j), Xall(j), gamma_start(j), gg(j), module_prior_par(j), binary?true:false, 0); // no mcmc
      logliks(j) = modules[j].loglik;
      proposed_logliks(j) = proposed_modules[j].loglik;
    }
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        clog << m << " " << max(abs(z)) << endl;
      } 
    }
  }
  //clog << "tot accepted: " << accepted << endl;
  
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
  //if(isnan(result)){
  //  cout << y.t() << endl << mean_post.t() << endl << inv_var_post << endl << a << " " << b << endl;
  //  cout << arma::conv_to<double>::from(y.t()*y - mean_post.t() * inv_var_post * mean_post) << endl;
  //  throw 1;
  //}
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
  double normconst = - p/2.0 * log(2*M_PI) + .5 * bdet(Si);
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
  //Xfull = arma::zeros(0,0);
  J_pre = arma::zeros(0,0);
  J_now = arma::zeros(0,0);
  mean_pre = arma::zeros(0);
  var_pre = arma::zeros(0,0);
  inv_var_pre = arma::zeros(0,0);
  //stretcher = arma::zeros(0,0);
  inv_var_post = arma::zeros(0,0);
  var_post = arma::zeros(0,0);
  mean_post = arma::zeros(0);
  beta_post = 0.0;
  sigmasq_sample = 0.0;
  theta_sample = arma::zeros(0);
  xb_mean = arma::zeros(0);
  xb_sample = arma::zeros(0);
  Jcs = arma::zeros(0);
  
  //theta_p_mean = arma::zeros(0);
  theta_p_sample = arma::zeros(0);
  ej_next = arma::zeros(0);
  //marglik_module_var = arma::zeros(0,0);
  //marglik_integr_var = arma::zeros(0,0);
  //marglik_mean = arma::zeros(0);
  //bigsigma = arma::zeros(0,0);
  Px = arma::zeros(0,0);
  //KsKs = arma::zeros(0,0);
  //KsK = arma::zeros(0,0);
}


Module::Module(arma::mat& X_full, arma::vec& e, arma::mat& X, double g_prior,
               arma::mat& Jpre, arma::mat& Jnow, arma::vec& mean_post_pre, arma::vec& theta_sample_pre,
               arma::vec& ingrid, arma::vec& splits, double fixed_sigma=-1,
               double ain=2.1, double bin=1.1){
  //cout << "ModularLinReg :: module 0 " << endl;
  //Xfull = X_full;
  
  ej = e;
  Xj = X;
  n = e.n_elem;
  pj = Xj.n_cols;
  
  icept = arma::mean(ej);
  xs = bmfuncs::col_sums(Xj.t());
  //ej = ej - icept;
  bmean = arma::conv_to<double>::from(arma::inv_sympd(xs.t()*xs)*xs.t()*ej);
  //ej = ej - xs*bmean;
  
  if(fixed_sigma == -1){
    fix_sigma = false;
  } else {
    fix_sigma = true;
  }
  
  // how much of a coarsening each variable of this module is
  //Jcs = bmfuncs::col_sums(Jnow);
  // prior variance = gres: g Jcs   ie large variance if many regressors are being coarsened, small otherwise
  //gprior: g(X'X)^-1  
  grid = ingrid;
  g = g_prior;
  msplits = splits;
  J_now = Jnow;
  ridge = 0.1;
  XtX = Xj.t() * Xj + ridge*arma::eye(Xj.n_cols, Xj.n_cols);
  XtXi = arma::inv_sympd(XtX); 
  mean_pre = arma::zeros(Xj.n_cols);
  var_pre = arma::zeros(0,0); //g*XtXi; //
  inv_var_post = (g+1.0)/g * XtX; //var_pre + XtX; //
  
  //clog << var_pre << endl;
  var_post = g/(g+1.0) * (XtXi); // arma::inv_sympd(inv_var_post);  //
  mean_post = var_post * Xj.t() * (ej - icept - xs*bmean);
  In = arma::eye(n,n);
  a = ain; // parametrization: a = mean^2 / variance + 2
  b = bin;  //                  b = (mean^3 + mean*variance) / variance
  Px = Xj * var_post * Xj.t();  //Xj * XtXi * Xj.t();
  if(!fix_sigma){
    arma::vec yy = ej - icept - xs*bmean;
    beta_post = arma::conv_to<double>::from(.5*(yy.t() * (In - g/(g+1.0) * Px) * yy));
    
    sigmasq_sample = 1.0/bmrandom::rndpp_gamma(a + n/2.0, 1.0/(b + beta_post));
    sigmasq_mean = (b+beta_post)/(a+n/2.0+1);
  } else {
    sigmasq_sample = fixed_sigma;//1.0;
  }
  
  theta_sample = bmrandom::rndpp_mvnormal2(1, mean_post + bmean, sigmasq_sample*var_post).t();
  
  theta_p_sample = (Jpre * theta_sample_pre + Jnow * theta_sample);
  
  
  xb_sample = Xj * theta_sample + icept;
  ej_next = ej - xb_sample;
  
  xb_mean = Xj * (bmean + mean_post) + icept;
}

void Module::redo(const arma::vec& e,
                  const arma::mat& Jpre, 
                  const arma::mat& Jnow,
                  const arma::vec& theta_sample_pre){
  ej = e;
  cout << "redoing module" << endl;
  icept = arma::mean(ej);
  xs = bmfuncs::col_sums(Xj.t());
  //ej = ej - icept;
  bmean = arma::conv_to<double>::from(arma::inv_sympd(xs.t()*xs)*xs.t()*ej);
  //ej = ej - xs*bmean;
  
  if(!fix_sigma){
    arma::vec yy = ej - icept - xs*bmean;
    beta_post = arma::conv_to<double>::from(.5*(yy.t() * (In - g/(g+1.0) * Px) * yy));
    
    cout << "beta post " << beta_post << endl;
    sigmasq_sample = 1.0/bmrandom::rndpp_gamma(a + n/2.0, 1.0/(b + beta_post)); //1.0/rndpp_gamma(n/2.0, 1.0/beta_post);
  } else {
    //sigmasq_sample = fix_sigma;
  }
  mean_post = var_post * Xj.t() * (ej - icept - xs*bmean);
  cout << "sigmasq sample " << sigmasq_sample << endl;
  
  theta_sample = bmrandom::rndpp_mvnormal2(1, bmean + mean_post, sigmasq_sample*var_post).t();
  theta_p_sample = (Jpre * theta_sample_pre + Jnow * theta_sample);
  
  
  xb_sample = Xj * theta_sample + icept;
  ej_next = ej - xb_sample;
  xb_mean = Xj * (bmean + mean_post) + icept;
  
  //if(MCMCSWITCH == 1){
  //marglik_module_var = sigmasq_sample * (Xj * var_post * Xj.t()); 
  //integrated ml variance to be used with previous module_vars
  //marglik_integr_var = sigmasq_sample * (In + Xj * var_pre * Xj.t());
  //}
  //bigsigma = sigmasq_sample * var_post;
  cout << "redone module " << endl;
}


ModularLinReg::ModularLinReg(const arma::vec& yin, 
                             const arma::mat& Xin, 
                             double g, 
                             const arma::field<arma::vec>& in_splits, 
                             double radius, int set_max_stages, 
                             double fixed_sigma=-1, bool fixed_grids = true,
                             double ain=2.1, double bin=1.1,
                             double spar=1.0){
  
  
  rad = radius;
  
  a = ain;
  b = bin;
  fixed_splits = fixed_grids;
  fix_sigma = fixed_sigma==-1? false : true;
  nested = true;
  
  structpar = spar;
  
  fixed_sigma_v = fixed_sigma==-1? -1 : fixed_sigma;
  
  
  y = yin-arma::mean(yin);
  X = Xin;
  
  xb = arma::zeros(y.n_elem);
  
  max_stages = set_max_stages;
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
  cout << "? 0 " << endl;
  bigsplit(0) = split_seq(0);
  cumsplit(0) = bigsplit(0).n_elem;
  //check
  
  for(unsigned int s=1; (s<split_seq.n_elem); s++){
    if(split_seq(s).n_elem>0){
      n_stages ++;
      bigsplit(s) = arma::join_vert(bigsplit(s-1), split_seq(s));
      bigsplit(s) = arma::unique(bigsplit(s));
      cumsplit(s) = bigsplit(s).n_elem;
      
    } else {
      break;
    }
  }
  cout << "? 1 " << endl;
  // initialization of structural variables
  J_field = arma::field<arma::mat>(max_stages); //n_stages+1
  X_field = arma::field<arma::mat>(max_stages); //J_field.n_elem
  //mod_reshaper = arma::field<arma::mat>(max_stages-1);
  
  // initialization of prior variables
  //m_field = arma::field<arma::vec>(max_stages);
  //M_field = arma::field<arma::mat>(max_stages);
  mu_field = arma::field<arma::vec>(max_stages);
  the_sample_field = arma::field<arma::vec>(max_stages);
  
  // structure of module 0
  grid = arma::linspace(0, p-1, p);
  pones = arma::ones(p, 1);
  cout << "? 2 " << endl;
  cout << split_seq(0) << endl;
  
  J_field(0) = bmfuncs::multi_split(pones, split_seq(0), p);
  //J_field(0) = wavelettize(J_field(0));
  if(rad != 0){
    J_field(0) = bmfuncs::J_smooth(J_field(0), rad * (n_stages-0), nested);
  }
  
  X_field(0) = X * J_field(0);
  //
  theta_p_scales = arma::field<arma::vec>(max_stages);
  sigmasq_scales = arma::zeros(max_stages);
  //bigsigma = arma::field<arma::mat>(max_stages);
  //bigsigma_of_incr = arma::field<arma::mat>(max_stages);
  cout << "? 3 " << endl;
  // structure of all other modules
  cout << "ModularLinReg :: creating other elements after 0 " << endl;
  
  for(unsigned int j=1; j<n_stages; j++){
    if(nested){
      J_field(j) = bmfuncs::multi_split(pones, bigsplit(j), p);
    } else {
      arma::mat bigmat = bmfuncs::multi_split(pones, bigsplit(j-1), p);
      J_field(j) = bmfuncs::multi_split_nonnested(bigmat, bmfuncs::bmms_setdiff(bigsplit(j), bigsplit(j-1)), p);
    }
    if(rad != 0){
      J_field(j) = bmfuncs::J_smooth(J_field(j), rad * (n_stages-j), nested);
    }
    X_field(j) = X * J_field(j);
  }
  
  
  // e, X, Jpre, Jnow, prior mean and var, previously sampled theta
  faux_J_previous = arma::zeros(J_field(0).n_rows);
  faux_theta_previous_sample = arma::zeros(1);
  
  cout << "-adding first module " << fixed_sigma_v << endl;
  Module adding_module(X, y, X_field(0), g_prior, faux_J_previous, J_field(0), 
                       faux_theta_previous_sample, faux_theta_previous_sample, grid, 
                       bigsplit(0), fixed_sigma_v, a, b);
  cout << "added." << endl;
  //current
  modules.push_back(adding_module);
  xb += modules[0].xb_mean;
  
  mu_field(0) = modules[0].J_now * modules[0].mean_post;
  the_sample_field(0) = modules[0].J_now * modules[0].theta_sample;
  
  theta_p_scales(0) = modules[0].theta_p_sample;
  sigmasq_scales(0) = modules[0].sigmasq_sample;
  //bigsigma(0) = modules[0].bigsigma;
  //bigsigma_of_incr(0) = modules[0].bigsigma;
  
  for(unsigned int s=1; s<n_stages; s++){
    //cout << "-adding other module" << endl;
    adding_module = Module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), 
                           J_field(s-1), J_field(s), 
                           modules[s-1].mean_post, modules[s-1].theta_sample,
                           grid, bigsplit(s), fixed_sigma_v, a, b);
    //cout << "--added." << endl;
    modules.push_back(adding_module);
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = mu_field(s-1) + modules[s].J_now * modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    xb += modules[s].xb_mean;
    
    //mod_reshaper(s-1) = reshaper(J_field, s);
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
  }
  
  intercept = arma::mean(y - X*the_sample_field(n_stages-1));
  
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        // integrating out coeff, N(0, (I + XVX')) = N(0, (I - XV*X')^-1) // function here takes precision mat
        loglik(s) = modular_loglikn(modules[s].ej, 1.0/modules[s].sigmasq_sample * (In - modules[s].Px));
        //modular_loglikn(modules[s].ej_next, 1.0/modules[s].sigmasq_sample * In)
      }
    } else {
      double sigprior = 0;
      for(int s=0; s<n_stages; s++){
        loglik(s) = modular_loglik2(modules[s].ej, modules[s].mean_post, modules[s].inv_var_post,
               modules[s].a, modules[s].b) + sigprior;
        sigprior += R::dgamma(1.0/sigmasq_scales(s), modules[s].a, 1.0/modules[s].b, 1);
        cout << "loglik " << loglik(s) << endl;
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
  bigsplit(s) = arma::unique(bigsplit(s));
  cumsplit(s) = bigsplit(s).n_elem;
  
  int j=s;
  if(nested){
    J_field(j) = bmfuncs::multi_split(pones, bigsplit(j), p);
  } else {
    arma::mat bigmat = bmfuncs::multi_split(pones, bigsplit(j-1), p);
    J_field(j) = bmfuncs::multi_split_nonnested(bigmat, bmfuncs::bmms_setdiff(bigsplit(j), bigsplit(j-1)), p);
  }
  
  if(rad != 0){
    J_field(s) = bmfuncs::J_smooth(J_field(s), rad * (n_stages-s), nested);
  }
  X_field(s) = X * J_field(s);
  
  Module adding_module = Module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), 
                                J_field(s-1), J_field(s), 
                                modules[s-1].mean_post, modules[s-1].theta_sample,
                                grid, bigsplit(s), fixed_sigma_v, a, b);
  modules.push_back(adding_module);
  xb += modules[s].xb_mean;
  
  theta_p_scales(s) = modules[s].theta_p_sample;
  mu_field(s) = mu_field(s-1) + modules[s].J_now * modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
  the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
  sigmasq_scales(s) = modules[s].sigmasq_sample;
  
  intercept = arma::mean(y - X*the_sample_field(n_stages-1));
  
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
        // integrating out coeff, N(0, (I + XVX')) = N(0, (I - XV*X')^-1) // function here takes precision mat
        loglik(s) = modular_loglikn(modules[s].ej, 1.0/modules[s].sigmasq_sample * (In - modules[s].Px));
        //modular_loglikn(modules[s].ej_next, 1.0/modules[s].sigmasq_sample * In)
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
  //m_field(n_stages) = arma::zeros(0);
  //M_field(n_stages) = arma::zeros(0,0);
  xb -= modules[n_stages].xb_mean;
  modules.pop_back();
  
  intercept = arma::mean(y - X*the_sample_field(n_stages-1));
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
        // integrating out coeff, N(0, (I + XVX')) = N(0, (I - XV*X')^-1) // function here takes precision mat
        loglik(s) = modular_loglikn(modules[s].ej, 1.0/modules[s].sigmasq_sample * (In - modules[s].Px));
        //modular_loglikn(modules[s].ej_next, 1.0/modules[s].sigmasq_sample * In)
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
  modules[0] = Module(X, y, X_field(0), g_prior, faux_J_previous, J_field(0), 
                      faux_theta_previous_sample, faux_theta_previous_sample, grid, 
                      bigsplit(0), fixed_sigma_v, a, b);
  
  mu_field(0) = modules[0].J_now * modules[0].mean_post; //modules[0].J_now * modules[0].mean_post;
  the_sample_field(0) = modules[0].J_now * modules[0].theta_sample;
  xb = modules[0].xb_mean;
  theta_p_scales(0) = modules[0].theta_p_sample;
  sigmasq_scales(0) = modules[0].sigmasq_sample;
  //bigsigma(0) = modules[0].bigsigma;
  //bigsigma_of_incr(0) = modules[0].bigsigma;
  
  for(unsigned int s=1; s<n_stages; s++){
    cout << "-adding other module" << endl;
    modules[s].redo(modules[s-1].ej_next, modules[s-1].J_now, modules[s].J_now, modules[s-1].theta_sample);
    cout << "--added." << endl;
    
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = mu_field(s-1) + modules[s].J_now * modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    xb += modules[s].xb_mean;
  }
  
  intercept = arma::mean(y - X*the_sample_field(n_stages-1));
  
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        // integrating out coeff, N(0, (I + XVX')) = N(0, (I - XV*X')^-1) // function here takes precision mat
        loglik(s) = modular_loglikn(modules[s].ej, 1.0/modules[s].sigmasq_sample * (In - modules[s].Px));
        //modular_loglikn(modules[s].ej_next, 1.0/modules[s].sigmasq_sample * In)
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
      bigsplit(s) = arma::unique(bigsplit(s));
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
  //m_field = arma::field<arma::vec>(max_stages);
  //M_field = arma::field<arma::mat>(max_stages);
  mu_field = arma::field<arma::vec>(max_stages);
  the_sample_field = arma::field<arma::vec>(max_stages);
  
  // structure of module 0
  //grid = arma::linspace(0, p-1, p);
  //pones = arma::ones(p, 1);
  
  J_field(0) = bmfuncs::multi_split(pones, split_seq(0), p);
  //J_field(0) = wavelettize(J_field(0));
  if(rad != 0){
    J_field(0) = bmfuncs::J_smooth(J_field(0), rad * (n_stages-0), nested);
  }
  X_field(0) = X * J_field(0);
  
  //
  theta_p_scales = arma::field<arma::vec>(max_stages);
  sigmasq_scales = arma::zeros(max_stages);
  //bigsigma = arma::field<arma::mat>(max_stages);
  //bigsigma_of_incr = arma::field<arma::mat>(max_stages);
  
  // structure of all other modules
  //cout << "ModularLinReg :: creating other elements after 0 " << endl;
  for(unsigned int j=1; j<n_stages; j++){
    if(nested){
      J_field(j) = bmfuncs::multi_split(pones, bigsplit(j), p);
    } else {
      arma::mat bigmat = bmfuncs::multi_split(pones, bigsplit(j-1), p);
      J_field(j) = bmfuncs::multi_split_nonnested(bigmat, bmfuncs::bmms_setdiff(bigsplit(j), bigsplit(j-1)), p);
    }
    //J_field(j) = wavelettize(J_field(j));
    if(rad != 0){
      J_field(j) = bmfuncs::J_smooth(J_field(j), rad * (n_stages-j), nested);
    }
    X_field(j) = X * J_field(j);
  }
  
  // e, X, Jpre, Jnow, prior mean and var, previously sampled theta
  faux_J_previous = arma::zeros(J_field(0).n_rows);
  faux_theta_previous_sample = arma::zeros(1);
  
  cout << "-adding first module [change_all] " << fixed_sigma_v << endl;
  Module adding_module(X, y, X_field(0), g_prior, faux_J_previous, J_field(0), 
                       faux_theta_previous_sample, faux_theta_previous_sample, grid, bigsplit(0), fixed_sigma_v, a, b);
  cout << "added." << endl;
  modules.push_back(adding_module);
  
  mu_field(0) = modules[0].J_now * modules[0].mean_post;
  the_sample_field(0) = modules[0].J_now * modules[0].theta_sample;
  xb = modules[0].xb_mean;
  theta_p_scales(0) = modules[0].theta_p_sample;
  sigmasq_scales(0) = modules[0].sigmasq_sample;
  //bigsigma(0) = modules[0].bigsigma;
  //bigsigma_of_incr(0) = modules[0].bigsigma;
  
  for(unsigned int s=1; s<n_stages; s++){
    //cout << "-adding other module" << endl;
    adding_module = Module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), 
                           J_field(s-1), J_field(s), 
                           modules[s-1].mean_post, modules[s-1].theta_sample,
                           grid, bigsplit(s), false, a, b);
    //cout << "--added." << endl;
    modules.push_back(adding_module);
    
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = mu_field(s-1) + modules[s].J_now * modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    xb += modules[s].xb_mean;
    //mod_reshaper(s-1) = reshaper(J_field, s);
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
  }
  intercept = arma::mean(y - X*the_sample_field(n_stages-1));
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        // integrating out coeff, N(0, (I + XVX')) = N(0, (I - XV*X')^-1) // function here takes precision mat
        loglik(s) = modular_loglikn(modules[s].ej, 1.0/modules[s].sigmasq_sample * (In - modules[s].Px));
        //modular_loglikn(modules[s].ej_next, 1.0/modules[s].sigmasq_sample * In)
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
  
  for(unsigned int s=whichone; s<n_stages; s++){ 
    //starting from whichone, but n_stages is the same here
    
    bigsplit(s) = arma::join_vert(bigsplit(s-1), split_seq(s));
    bigsplit(s) = arma::unique(bigsplit(s));
    cumsplit(s) = bigsplit(s).n_elem;
    
    int j=s;
    if(nested){
      J_field(j) = bmfuncs::multi_split(pones, bigsplit(j), p);
    } else {
      arma::mat bigmat = bmfuncs::multi_split(pones, bigsplit(j-1), p);
      J_field(j) = bmfuncs::multi_split_nonnested(bigmat, bmfuncs::bmms_setdiff(bigsplit(j), bigsplit(j-1)), p);
    }
    if(rad != 0){
      J_field(s) = bmfuncs::J_smooth(J_field(s), rad * (n_stages-s), nested);
    }
    X_field(s) = X * J_field(s);
    
    xb -= modules[s].xb_mean;
    Module adding_module(X, modules[s-1].ej_next, X_field(s), pow(g_prior, 1.0/(s+1.0)), 
                         J_field(s-1), J_field(s), 
                         modules[s-1].mean_post, modules[s-1].theta_sample, grid, bigsplit(s), fixed_sigma_v, a, b);
    modules[s] = adding_module;
    theta_p_scales(s) = modules[s].theta_p_sample;
    mu_field(s) = mu_field(s-1) + modules[s].J_now * modules[s].mean_post;//mu_field(s-1) + modules[s].J_now * modules[s].mean_post;
    the_sample_field(s) = the_sample_field(s-1) + modules[s].J_now * modules[s].theta_sample;
    sigmasq_scales(s) = modules[s].sigmasq_sample;
    xb += modules[s].xb_mean;
    //mod_reshaper(s-1) = reshaper(J_field, s);
    //bigsigma(s) = (pow(1.0/(n+1), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    //bigsigma_of_incr(s) = (pow(n/(n+1.0), 2) * mod_reshaper(s-1) * modules[s-1].bigsigma * mod_reshaper(s-1).t() + 
    //  modules[s].bigsigma);
    
  }
  intercept = arma::mean(y - X*the_sample_field(n_stages-1));
  if(!fixed_splits){
    if(fix_sigma){
      for(int s=0; s<n_stages; s++){ 
        // integrating out coeff, N(0, (I + XVX')) = N(0, (I - XV*X')^-1) // function here takes precision mat
        loglik(s) = modular_loglikn(modules[s].ej, 1.0/modules[s].sigmasq_sample * (In - modules[s].Px));
        //modular_loglikn(modules[s].ej_next, 1.0/modules[s].sigmasq_sample * In)
        //clog << "ll " << loglik(s) << endl;
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
  double ck = lambda_prop * pow(2.0, ss);
  double proposal = (- ck * tot_split_prop * std::log(tot_split_prop));
  double orig = (- ck * tot_split_orig * std::log(tot_split_orig));
  return exp(proposal - orig);
}


double totsplit_prior_ratio(int tot_split_prop, int tot_split_orig, int norp, int ss, double lambda_prop){
  //lambda_prop is 1/variance;
  double means = pow(2, ss);
  return exp(-lambda_prop/2.0 * pow(tot_split_prop - means, 2) + lambda_prop/2.0 * pow(tot_split_orig - means, 2) ); //prior_ratio;
}

double split_struct_ratio2(const arma::field<arma::vec>& proposed, const arma::field<arma::vec>& original,
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
  
  arma::vec proposed_unique = arma::unique(proposed(stage_proposed));
  arma::vec original_unique = arma::unique(original(stage_original));
  
  arma::vec proposed_here_diff = arma::diff(arma::join_vert(proposed_unique, arma::ones(1)*p));
  arma::vec original_here_diff = arma::diff(arma::join_vert(original_unique, arma::ones(1)*p));
  
  cout << "SSRATIO CALC" << endl << proposed_here_diff.t() << endl << original_here_diff.t() << endl;
  double rat = 1.0;
  try{
    /*
     double minprop = proposed_here_diff.min();
     if(minprop == 0){
     clog << "Runtime error: diff in splits=0 hence wrong move." << endl;
     throw 1;
     }
     double minorig = original_here_diff.min();
     if(minorig == 0){
     clog << "Runtime error: diff in splits=0 hence wrong move." << endl;
     throw 1;
     }*/
    //if(minprop == 1){
    //  return 0;
    //}
    rat = pow(arma::accu(1.0/pow(original_here_diff, 2))/arma::accu(1.0/pow(proposed_here_diff, 2)), param); //pow(log(1.0+minprop)/log(1.0+minorig), param);
    //clog << "min prop diff " << proposed_here_diff.min() << ". ratio with orig=" << rat << endl;
    //clog << "alternative " << arma::accu(1.0/pow(original_here_diff, 2))/arma::accu(1.0/pow(proposed_here_diff, 2)) << endl << endl;
    //clog << "ratio " << rat << " mp:" << minprop << " mo:" << minorig << endl;
    return rat;
  } catch(...) {
    // min has no elements error -- happens when proposing drop from 2 to 1.
    throw 1;
  }
  
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


VarSelOps::VarSelOps(const arma::vec& yy, const arma::mat& XX, const arma::vec& prior,
                     double gin=-1.0, double model_prior_par_in=1, bool fix_sigma=false, int iter=0){
  //clog << "creating " << endl;
  y = yy;
  X = XX;
  mcmc = iter;
  
  p = X.n_cols;
  arma::vec p_indices = arma::linspace<arma::vec>(0, p-1, p);
  n = y.n_elem;
  
  fixsigma = fix_sigma;
  model_prior_par = model_prior_par_in;
  g = gin;
  
  //clog << "initing " << endl;
  icept_stored = arma::zeros(mcmc==0? 1: mcmc);
  beta_stored = arma::zeros(p, mcmc==0? 1: mcmc);
  gamma_stored = arma::zeros(p, mcmc==0? 1: mcmc);
  sigmasq_stored = arma::zeros(mcmc==0? 1: mcmc);
  
  gamma = arma::uvec(p);
  for(int j=0; j<p; j++){
    gamma(j) = 1*bmrandom::boolbern(prior(j));
  }
  
  
  //clog << "BS2 " << endl;
  gammaix = arma::find(gamma);
  //clog << "test  1" << endl;
  sampled_model = bmmodels::BayesLMg(y, X.cols(gammaix), g, true, fixsigma);
  
  loglik = sampled_model.marglik - model_prior_par*sampled_model.p;
  
  
  //clog << "lin pred " << endl;
  // only ok for mcmc=1
  linear_predictor = X.cols(gammaix) * sampled_model.b + sampled_model.icept;
}

void VarSelOps::forward(int ix){
  // changes a single variable
  mcmc = 1;
  int m = 0;
  //clog << "init forward " << endl;
  //clog << ix << endl;
  gamma_proposal = gamma;
  //clog << gamma_proposal << endl;
  gamma_proposal(ix) = 1-gamma(ix);
  //clog << gamma_proposal << endl;
  arma::uvec gammaix_proposal = arma::find(gamma_proposal);
  //clog << "changing X " << endl;
  gammaix = gammaix_proposal;
  gamma = gamma_proposal;
  
  //clog << "sampling " << endl;
  sampled_model = bmmodels::BayesLMg(y, X.cols(gammaix), g, true, fixsigma);
  
  loglik = sampled_model.marglik - model_prior_par*sampled_model.p;
  //clog << "forward: " << loglik << endl;
  
  arma::vec beta_full = arma::zeros(p);
  beta_full.elem(gammaix) = sampled_model.b;
  icept_stored(m) = sampled_model.icept;
  beta_stored.col(m) = beta_full;
  gamma_stored.col(m) = arma::conv_to<arma::vec>::from(gamma);
  sigmasq_stored(m) = sampled_model.sigmasq;
  //clog << "linear pred " << endl;
  linear_predictor = X.cols(gammaix) * sampled_model.b + sampled_model.icept;
}

void VarSelOps::change_y(const arma::vec& newy){
  int m = 0;
  // no sampling of gamma, keep everything as is, recalculate logposterior
  //clog << "changing y " << endl;
  y = newy;
  n = y.n_elem;
  
  //clog << "sampling  " << endl;
  sampled_model = bmmodels::BayesLMg(y, X.cols(gammaix), g, false, fixsigma); // false = no sampling, just take the logpost
  loglik = sampled_model.marglik - model_prior_par*sampled_model.p;
  
  //arma::vec beta_full = arma::zeros(p);
  //beta_full.elem(gammaix) = sampled_model.b;
  
  //icept_stored(m) = sampled_model.icept;
  //beta_stored.col(m) = beta_full;
  //gamma_stored.col(m) = arma::conv_to<arma::vec>::from(gamma);
  //sigmasq_stored(m) = sampled_model.sigmasq;
  //clog << "finding lin pred " << endl;
  //linear_predictor = X.cols(gammaix) * sampled_model.b + sampled_model.icept;
}


