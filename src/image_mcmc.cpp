//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::interfaces(r)]]

#include "truncnormal_sample.h"
#include "image_mcmc_helper.h"

using namespace std;

/*
*              FUNCTIONS TO MOVE SPLIT LOCATIONS
*                   TO APPROPRIATE NEW LOCS
*/

// focuses around onesplit, with radius

arma::mat splitmask_focus(const arma::mat& mask_of_splits, arma::vec onesplit, int radius_int=1){
  int x1 = onesplit(0) - radius_int;
  int x2 = onesplit(0) + radius_int;
  int y1 = onesplit(1) - radius_int;
  int y2 = onesplit(1) + radius_int;
  int xlim = mask_of_splits.n_rows-1;
  int ylim = mask_of_splits.n_cols-1;
  x1 = max(0, x1);
  x2 = min(x2, xlim);
  y1 = max(0, y1);
  y2 = min(y2, ylim);
  
  // focusing
  arma::mat around_split = mask_of_splits.submat(x1, y1, x2, y2);
  return around_split;
}

// given focus of a split matrix, counts the zeros (ie number of available locations to move to)
int number_availables(const arma::mat& splitmask_focus){
  int c1 = arma::accu(1-splitmask_focus.elem(arma::find(splitmask_focus == 0)));
  return c1;
}


// given a mask of splits,
// move onesplit to another location around it (radius fixed)
// return new mask of splits
// also gives ratio: [number of moves TO proposed]/[number of moves FROM proposed]
// = [prob moving FROM proposed back]/[prob moving TO proposed forward]

arma::mat split_move2d(const arma::mat& mask_of_splits, 
                       const arma::mat& mask_nosplits,
                       arma::vec onesplit, 
                       double& to_from_ratio, int radius_int=1){
  // coordinates to focus on
  int x1 = onesplit(0) - radius_int;
  int x2 = onesplit(0) + radius_int;
  int y1 = onesplit(1) - radius_int;
  int y2 = onesplit(1) + radius_int;
  int xlim = mask_of_splits.n_rows-1;
  int ylim = mask_of_splits.n_cols-1;
  x1 = max(0, x1);
  x2 = min(x2, xlim);
  y1 = max(0, y1);
  y2 = min(y2, ylim);
  
  // focusing
  arma::mat around_split = mask_of_splits.submat(x1, y1, x2, y2);
  arma::mat around_nosplit = mask_nosplits.submat(x1, y1, x2, y2);
  //cout << "focusing " << endl << around_split << endl;
  // pick random location among available
  
  arma::uvec available_locations = arma::find((around_split+around_nosplit)==1);
  
  // count moves TO
  double avail_to = arma::accu(1-around_split.elem(available_locations));
  //cout << "number of moves TO new: " << avail_to << endl;
  
  // choose random location among available ones
  // in focused view
  arma::uvec rand_location = sample_index(1, available_locations.n_elem);
  rand_location = available_locations.elem(rand_location);
  
  // original location in focused coordinate system
  arma::vec orig_location = arma::zeros(2);
  orig_location(0) = onesplit(0) - x1;
  orig_location(1) = onesplit(1) - y1;
  
  // change values in the focused view
  around_split.elem(rand_location).fill(mask_of_splits(onesplit(0), onesplit(1)));
  around_split(orig_location(0), orig_location(1)) = 0;
  // substitute on original matrix
  arma::mat returning_mat = mask_of_splits;
  returning_mat.submat(x1, y1, x2, y2) = around_split;
  
  // the moved split has these coordinates
  arma::mat new_location = arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(around_split), rand_location));
  new_location(0) = new_location(0) + x1;
  new_location(1) = new_location(1) + y1;
  //cout << "new location in original coordinates : " << new_location.t() << endl;
  
  arma::mat avail_around_moved = splitmask_focus(returning_mat, new_location, radius_int);
  arma::mat avail_around_nosplit = splitmask_focus(returning_mat, new_location, radius_int);
  double avail_from = arma::accu(1-avail_around_moved.elem(arma::find((avail_around_moved+avail_around_nosplit) == 1)));
  //number_availables(avail_around_moved);
  //cout << "number of moves FROM new: " << avail_from << endl;
  
  to_from_ratio = avail_to/avail_from;
  return returning_mat;
}

//' Vector index to matrix subscripts
//' 
//' Get matrix subscripts from corresponding vector indices (both start from 0).
//' This is a utility function using Armadillo's ind2sub function.
//' @param index a vector of indices
//' @param m a matrix (only its size is important)
//' @export
//[[Rcpp::export]]
arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m){
  return arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(m), index));
}

arma::mat split_add2d(arma::mat mask_of_splits, 
                      arma::mat mask_nosplits,
                      int lev,
                      double& to_from_ratio){
  
  // mask_nosplits has 0 where a split should not be put
  // and 1 otherwise
  
  //mask_temp.elem(arma::find(mask_nosplits==0)).fill(-1);
  //clog << mask_temp << endl;
  // coordinates to focus on
  arma::uvec available_locations = arma::find((mask_of_splits+mask_nosplits)==1);
  
  // count moves TO
  double avail_to = arma::accu(1-mask_of_splits.elem(available_locations));
  //cout << "number of moves TO new: " << avail_to << endl;
  
  // choose random location among available ones
  // in focused view
  arma::uvec rand_location = sample_index(1, available_locations.n_elem);
  rand_location = available_locations.elem(rand_location);
  
  // change values in the focused view
  mask_of_splits.elem(rand_location).fill(lev+1);
  
  // occupied locations
  available_locations = arma::find(mask_of_splits>0);
  double avail_from = available_locations.n_elem;
  //cout << "number of moves FROM new: " << avail_from << endl;
  
  to_from_ratio = avail_to/avail_from;
  //mask_of_splits.elem(arma::find(mask_of_splits==-1)).fill(0);
  return mask_of_splits;
}

arma::mat stage_add2d(arma::mat mask_of_splits, 
                      arma::mat mask_nosplits,
                      int lev,
                      double& to_from_ratio){
  
  // mask_nosplits has 0 where a split should not be put
  // and 1 otherwise
  
  //mask_temp.elem(arma::find(mask_nosplits==0)).fill(-1);
  //clog << mask_temp << endl;
  // coordinates to focus on
  arma::uvec available_locations = arma::find((mask_of_splits+mask_nosplits)==1);
  
  // count moves TO
  double avail_to = arma::accu(1-mask_of_splits.elem(available_locations));
  //cout << "number of moves TO new: " << avail_to << endl;
  
  // choose random location among available ones
  // in focused view
  arma::uvec rand_location = sample_index(2, available_locations.n_elem);
  rand_location = available_locations.elem(rand_location);
  
  // change values in the focused view
  mask_of_splits.elem(rand_location).fill(lev+1);
  
  // occupied locations
  //available_locations = arma::find(mask_of_splits>0);
  double avail_from = 1.0;//available_locations.n_elem;
  //cout << "number of moves FROM new: " << avail_from << endl;
  
  to_from_ratio = avail_to/avail_from;
  //mask_of_splits.elem(arma::find(mask_of_splits==-1)).fill(0);
  return mask_of_splits;
}


arma::mat split_drop2d(arma::mat mask_of_splits, 
                       const arma::mat& mask_nosplits,
                       int lev,
                       double& to_from_ratio){
  
  //arma::mat mask_temp = mask_of_splits;
  //mask_temp.elem(arma::find(mask_nosplits==0)).fill(-1);
  
  //clog << mask_temp << endl;
  
  // coordinates to focus on
  arma::uvec available_locations = arma::find(mask_of_splits==lev+1);
  
  // count moves TO
  double avail_to = available_locations.n_elem;
  
  //cout << "number of moves TO new: " << avail_to << endl;
  // choose random location among available ones
  arma::uvec rand_location = sample_index(1, available_locations.n_elem);
  rand_location = available_locations.elem(rand_location);
  
  // change values in the focused view
  arma::uvec coord = arma::ind2sub(arma::size(mask_nosplits), rand_location);
  //clog << coord << endl;
  mask_of_splits(coord(0), coord(1)) = 0.0;
  //clog << mask_temp << endl;
  
  // unoccupied locations
  available_locations = arma::find((mask_of_splits + mask_nosplits)==1);
  double avail_from = arma::accu(1-mask_of_splits.elem(available_locations));
  //cout << "number of moves FROM new: " << avail_from << endl;
  
  to_from_ratio = avail_to/avail_from;
  //mask_temp.elem(arma::find(mask_nosplits==0)).fill(0);
  return mask_of_splits;
}


void stage_drop2d(arma::mat mask_of_splits, 
                  const arma::mat& mask_nosplits,
                  int lev,
                  double& to_from_ratio){
  // the drop stage doesnt actually make a proposal but just calculates the ratio
  
  //arma::field<arma::mat> splitsub = splitmask_to_splitsub(mask_of_splits);
  //arma::field<arma::mat> new_splitsub = arma::field<arma::mat>(splitsub.n_elem-1);
  //for(unsigned int j = 0; j<new_splitsub.n_elem; j++){
  //  new_splitsub(j) = splitsub(j);
  //}
  
  // coordinates to focus on
  arma::uvec available_locations = arma::find(mask_of_splits==lev+1);
  
  // count moves TO
  double avail_to = 1.0;//available_locations.n_elem;
  
  // unoccupied locations
  available_locations = arma::find((mask_of_splits + mask_nosplits)==1);
  double avail_from = 2+arma::accu(1-mask_of_splits.elem(available_locations));
  //cout << "number of moves FROM new: " << avail_from << endl;
  to_from_ratio = avail_to/avail_from;
  //mask_temp.elem(arma::find(mask_nosplits==0)).fill(0);
  //return new_splitsub;
}


arma::mat proposal_move(const arma::mat& current_split_mask, 
                        const arma::mat& mask_nosplits,
                        const arma::field<arma::mat>& current_splits, 
                        int which_lev,
                        int which_split, 
                        int radius){
  
  double to_from_ratio = 0.0;
  arma::mat proposed_split_mask = split_move2d(current_split_mask, 
                                               mask_nosplits,
                                               current_splits(which_lev).row(which_split).t(), to_from_ratio, radius);
  
  cout << "proposed split mask " << endl << proposed_split_mask << endl;
  cout << "to_from_ratio " << to_from_ratio << endl;
  return proposed_split_mask;
}



// finally try building a model from scratch and output beta
//'@export
//[[Rcpp::export]]
Rcpp::List soi_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> splits,
                      arma::mat mask_forbid,
                      double lambda_centers, double lambda_ridge, int mcmc, int burn, 
                      int radius=2,
                      int start_movinglev=0,
                      int partnum=0, bool save=true,
                      bool save_splitmask = false){
  //former name: model_test
  
  //arma::field<arma::mat> old_splits,
  // vectorize X using splits
  // build blr model
  // propose split changes [move]
  // eval mhr & accept/reject
  // reconstruct & save beta at this step of the chain
  // start_movinglev = to fix hemispheres as first level
  int n = y.n_elem;
  double to_from_ratio = 0.0;
  //int radius = 2;
  double mhr = 0.0;
  
  int add_proposed = 0;
  int add_accepted = 0;
  int drop_proposed = 0;
  int drop_accepted = 0;
  
  int splitpar_proposed = 0;
  int splitpar_accepted = 0;
  
  int max_stages = splits.n_elem;
  
  //splits = merge_splits(splits, old_splits);
  
  //clog << splits << endl;
  arma::vec splitsparam_mcmc(mcmc-burn);
  arma::vec lambda_mcmc(mcmc-burn);
  arma::field<arma::cube> theta_mcmc(mcmc-burn);
  arma::field<arma::field<arma::mat>> splitsub_mcmc(mcmc-burn);
  arma::vec dim_mcmc(mcmc-burn);
  arma::cube splitmask_mcmc = arma::zeros(X.n_rows, X.n_cols, mcmc-burn);
  
  // initialize
  arma::mat propose_splitmask;
  arma::field<arma::mat> propose_splitsub;
  
  clog << "initalize " << endl;
  ModularLR2D bmms_t = ModularLR2D(y, X, splits, mask_forbid, max_stages, lambda_ridge);
  int num_levs = bmms_t.n_stages;
  //clog << "effective dim " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << endl;
  
  for(unsigned int m = 0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    
    if(m==0){ 
      clog << "starting mcmc" << endl;
    }
    int choices = 4;
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    
    int rnd_moving_lev = arma::randi<int>(arma::distr_param(start_movinglev, num_levs-1));
    
    if(move_type == 0){
      //clog << "R"; mh for lambda
      double lambdamult = arma::randn<double>();
      double new_lambda = 0.01 + lambda_ridge * exp(lambdamult);
      
      propose_splitsub = splitmask_to_splitsub(bmms_t.modules[bmms_t.n_stages-1].splitmask);
      ModularLR2D proposed_bayeslm(y, X, propose_splitsub, mask_forbid, max_stages, new_lambda);
      
      mhr = exp(arma::accu(proposed_bayeslm.logliks) - arma::accu(bmms_t.logliks)) * 
        exp(gammaprior_mhr(new_lambda, lambda_ridge));
      mhr = mhr > 1 ? 1 : mhr;
      
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        lambda_ridge = new_lambda;
        bmms_t = proposed_bayeslm;
      } else {
        bmms_t = ModularLR2D(y, X, propose_splitsub, mask_forbid, max_stages, lambda_ridge);
      }
      
      
    }
    if(move_type == 1){
      //clog << "M";
      
      //clog << "randomly moving L=" << rnd_moving_lev << endl;
      int num_splits = bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).n_rows;
      if ( num_splits == 0) {
        clog << bmms_t.modules[bmms_t.n_stages-1].splitmat << endl;
      }
      int rnd_moving_split = arma::randi<int>(arma::distr_param(0, num_splits-1));
      //clog << "randomly moving S=" << rnd_moving_split << endl;
      propose_splitmask = split_move2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                       bmms_t.mask_nosplits,
                                       bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).row(rnd_moving_split).t(), 
                                       to_from_ratio, radius);
      
      propose_splitsub = splitmask_to_splitsub(propose_splitmask);
      
      ModularLR2D proposed_bayeslm = bmms_t;
      
      //clog << "changing " << endl;
      proposed_bayeslm.change_module(rnd_moving_lev, propose_splitsub);
      //clog << "done" << endl;
      
      mhr = exp(arma::accu(proposed_bayeslm.logliks.subvec(rnd_moving_lev, proposed_bayeslm.n_stages-1)) - 
        arma::accu(bmms_t.logliks.subvec(rnd_moving_lev, bmms_t.n_stages-1))) * to_from_ratio;
      mhr = mhr > 1 ? 1 : mhr;
      
      //clog << "moving mhr " << mhr << endl << "to_from_ratio " << to_from_ratio << endl;
      
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        //clog << "[MOVE SPLIT " << stage << "] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
        bmms_t = proposed_bayeslm;
      } 
    }
    
    if(move_type == 2){
      //clog << "A";
      add_proposed ++;
      
      propose_splitmask = split_add2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                      bmms_t.mask_nosplits,
                                      rnd_moving_lev, 
                                      to_from_ratio);
      //cout << propose_splitmask << endl;
      propose_splitsub = splitmask_to_splitsub(propose_splitmask);
      //cout << propose_splitsub.row(rnd_moving) << endl;
      ModularLR2D proposed_bayeslm = bmms_t;
      proposed_bayeslm.change_module(rnd_moving_lev, propose_splitsub);
      
      //clog << rnd_moving_lev << " rows: " << bmms_t.modules[rnd_moving_lev].splitmat.n_rows << endl;
      double totsplit_prior_mhr = totsplit_prior_ratio(propose_splitsub(rnd_moving_lev).n_rows, 
                                                       bmms_t.modules[rnd_moving_lev].splitmat.n_rows, n, 
                                                       rnd_moving_lev, 
                                                       lambda_centers);
      
      mhr = exp(arma::accu(proposed_bayeslm.logliks.subvec(rnd_moving_lev, proposed_bayeslm.n_stages-1)) - 
        arma::accu(bmms_t.logliks.subvec(rnd_moving_lev, bmms_t.n_stages-1))) * 
        to_from_ratio * totsplit_prior_mhr;
      mhr = mhr > 1 ? 1 : mhr;
      //cout << mhr << endl;
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        //clog << "ADDED " << endl;
        add_accepted ++;
        bmms_t = proposed_bayeslm;
      } 
      //clog << "add " << totsplit_prior_mhr << endl;
    }
    if(move_type == 3){
      //clog << "D";
      int num_splits = bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).n_rows;
      
      if(num_splits>2){
        drop_proposed ++;
        //clog << "at level " << rnd_moving_lev << endl;
        propose_splitmask = split_drop2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                         bmms_t.mask_nosplits,
                                         rnd_moving_lev,  
                                         to_from_ratio);
        
        
        propose_splitsub = splitmask_to_splitsub(propose_splitmask);
        
        ModularLR2D proposed_bayeslm = bmms_t;
        //clog << "changing " << endl;
        proposed_bayeslm.change_module(rnd_moving_lev, propose_splitsub);
        //clog << "done" << endl;
        
        //clog << "seeing this " << endl;
        double totsplit_prior_mhr = totsplit_prior_ratio(propose_splitsub(rnd_moving_lev).n_rows, 
                                                         bmms_t.modules[rnd_moving_lev].splitmat.n_rows, n, 
                                                         rnd_moving_lev,  
                                                         lambda_centers);
        
        mhr = exp(arma::accu(proposed_bayeslm.logliks.subvec(rnd_moving_lev, proposed_bayeslm.n_stages-1)) - 
          arma::accu(bmms_t.logliks.subvec(rnd_moving_lev, bmms_t.n_stages-1))) * 
          to_from_ratio * totsplit_prior_mhr;
        mhr = mhr > 1 ? 1 : mhr;
        //clog << "and this " << endl;
        int accepted_proposal = rndpp_discrete({1-mhr, mhr});
        if(accepted_proposal == 1){
          drop_accepted ++;
          //clog << "DROPPED " << endl;
          bmms_t = proposed_bayeslm;
        }
        //clog << "drop " << totsplit_prior_mhr << endl;
        
      } else {
        //clog << "didnt" << endl;
      }
      
    }
    
    if(move_type == 4){
      splitpar_proposed++;
      //clog << "Q"; mh for split par
      double splitpar_m = arma::randn<double>();
      double new_lambda_centers = 0.1 + lambda_centers * exp(splitpar_m);
      
      mhr = exp(splitpar_prior(new_lambda_centers, bmms_t.modules[bmms_t.n_stages-1].splitmat.n_rows, n, max_stages-1) - 
        splitpar_prior(lambda_centers, bmms_t.modules[bmms_t.n_stages-1].splitmat.n_rows, n, max_stages-1)) * 
        exp(gammaprior_mhr(new_lambda_centers, lambda_centers));
      mhr = mhr > 1 ? 1 : mhr;
      
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        lambda_centers = new_lambda_centers;
        //clog << lambda_centers << endl;
        splitpar_accepted++;
      }
    }
    num_levs = bmms_t.n_stages;
    
    if(m>burn-1){
      int i = m-burn;
      theta_mcmc(i) = bmms_t.theta_sampled;
      splitsub_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].splitmat;
      lambda_mcmc(i) = bmms_t.lambda;
      splitsparam_mcmc(i) = lambda_centers;
      dim_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].effective_dimension;
      if(save_splitmask == true){
        splitmask_mcmc.slice(i) = bmms_t.modules[bmms_t.n_stages-1].splitmask;
      };
    }
    
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        Rcpp::checkUserInterrupt();
        clog << endl << 
          partnum << " " << floor(100.0*(m+0.0)/mcmc) << 
            " " << bmms_t.n_stages << " " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << 
              " " << add_accepted / (add_proposed+0.0) << " " << drop_accepted / (drop_proposed+0.0) << " " <<
                splitpar_accepted / (splitpar_proposed+0.0) << endl;
        if(save){
          bmms_t.modules[bmms_t.n_stages-1].splitmat.save("bmms_centers.temp");
        }
        
      } 
    }
    
  }
  
  return Rcpp::List::create(
    //Rcpp::Named("mu_post") = bayeslm.mu_post,
    //Rcpp::Named("Sigma_post") = bayeslm.Sigma_post,
    //Rcpp::Named("regions") = bayeslm.regions,
    Rcpp::Named("groupmask") = bmms_t.modules[bmms_t.n_stages-1].groupmask,
    Rcpp::Named("splitsub") = splitsub_mcmc,
    Rcpp::Named("dimension_overall") = dim_mcmc,
    Rcpp::Named("splitsparam") = splitsparam_mcmc,
    Rcpp::Named("lambda_ridge") = lambda_mcmc,
    Rcpp::Named("splitmask") = splitmask_mcmc,
    Rcpp::Named("sigmasq_post_mean") = bmms_t.modules[bmms_t.n_stages-1].sigmasq_post_mean,
    Rcpp::Named("a_post") = bmms_t.modules[bmms_t.n_stages-1].a_post,
    Rcpp::Named("b_post") = bmms_t.modules[bmms_t.n_stages-1].b_post,
    Rcpp::Named("theta_mc") = theta_mcmc,
    Rcpp::Named("sigmasq_sampled") = bmms_t.modules[bmms_t.n_stages-1].sigmasq_sampled
  );
}

//'@export
//[[Rcpp::export]]
Rcpp::List soi_binary_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> centers,
                    arma::mat mask_forbid,
                    double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius=2,
                    int start_movinglev=0,
                    int partnum=0, bool save=true,
                    bool save_splitmask = true,
                    bool fixsigma = false){
  //former name: model_test
  
  //arma::field<arma::mat> old_splits,
  // vectorize X using splits
  // build blr model
  // propose split changes [move]
  // eval mhr & accept/reject
  // reconstruct & save beta at this step of the chain
  // start_movinglev = to fix hemispheres as first level
  int n = y.n_elem;
  double to_from_ratio = 0.0;
  //int radius = 2;
  double mhr = 0.0;
  int max_stages = centers.n_elem;
  
  int add_proposed = 0;
  int add_accepted = 0;
  int drop_proposed = 0;
  int drop_accepted = 0;
  
  int splitpar_proposed = 0;
  int splitpar_accepted = 0;
  
  //splits = merge_splits(splits, old_splits);
  
  //clog << splits << endl;
  arma::vec splitsparam_mcmc(mcmc-burn);
  arma::vec lambda_mcmc(mcmc-burn);
  arma::field<arma::cube> theta_mcmc(mcmc-burn);
  arma::field<arma::vec> icept_mcmc(mcmc-burn);
  arma::field<arma::field<arma::mat>> splitsub_mcmc(mcmc-burn);
  arma::vec dim_mcmc(mcmc-burn);
  arma::cube splitmask_mcmc = arma::zeros(X.n_rows, X.n_cols, mcmc-burn);
  
  arma::field<arma::mat> Xflat_mcmc(mcmc-burn);
  arma::vec ybin(n);
  arma::vec z(n);
  arma::mat zsave(n, mcmc-burn);
  
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
  
  
  // initialize
  arma::mat propose_splitmask;
  arma::field<arma::mat> propose_splitsub;
  
  clog << "initalize " << endl;
  ModularLR2D bmms_t = ModularLR2D(y, X, centers, mask_forbid, max_stages, lambda_ridge, fixsigma);
  int num_levs = bmms_t.n_stages;
  
  ybin = y;
  z = bmms_t.modules[0].X_flat * bmms_t.modules[0].flatmodel.b;
  
  for(unsigned int m = 0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    if(m==0){ 
      clog << "starting mcmc" << endl;
    }
    int choices = 4;
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    
    int rnd_moving_lev = arma::randi<int>(arma::distr_param(start_movinglev, num_levs-1));
    
    if(move_type == 3){
      //clog << "R"; mh for lambda
      double lambdamult = arma::randn<double>();
      double new_lambda = 0.01 + lambda_ridge * exp(lambdamult);
      
      propose_splitsub = splitmask_to_splitsub(bmms_t.modules[bmms_t.n_stages-1].splitmask);
      ModularLR2D proposed_bayeslm(z, X, propose_splitsub, mask_forbid, max_stages, new_lambda, fixsigma);
      
      mhr = exp(arma::accu(proposed_bayeslm.logliks) - arma::accu(bmms_t.logliks)) * 
        exp(gammaprior_mhr(new_lambda, lambda_ridge));
      mhr = mhr > 1 ? 1 : mhr;
      
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        lambda_ridge = new_lambda;
        bmms_t = proposed_bayeslm;
      } else {
        bmms_t = ModularLR2D(z, X, propose_splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma);
      }
      
      
    }
    if(move_type == 0){
      //clog << "M";
      
      //clog << "randomly moving L=" << rnd_moving_lev << endl;
      int num_splits = bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).n_rows;
      if ( num_splits == 0) {
        clog << bmms_t.modules[bmms_t.n_stages-1].splitmat << endl;
      }
      int rnd_moving_split = arma::randi<int>(arma::distr_param(0, num_splits-1));
      //clog << "randomly moving S=" << rnd_moving_split << endl;
      propose_splitmask = split_move2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                       bmms_t.mask_nosplits,
                                       bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).row(rnd_moving_split).t(), 
                                       to_from_ratio, radius);
      
      propose_splitsub = splitmask_to_splitsub(propose_splitmask);
      
      ModularLR2D proposed_bayeslm = bmms_t;
      
      //clog << "changing " << endl;
      proposed_bayeslm.change_module(rnd_moving_lev, propose_splitsub);
      //clog << "done" << endl;
      
      mhr = exp(proposed_bayeslm.logliks(rnd_moving_lev) - bmms_t.logliks(rnd_moving_lev)) * to_from_ratio;
        //exp(arma::accu(proposed_bayeslm.logliks.subvec(rnd_moving_lev, proposed_bayeslm.n_stages-1)) - 
        //arma::accu(bmms_t.logliks.subvec(rnd_moving_lev, bmms_t.n_stages-1))) * to_from_ratio;
      mhr = mhr > 1 ? 1 : mhr;
      
      //clog << "moving mhr " << mhr << endl << "to_from_ratio " << to_from_ratio << endl;
      
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        //clog << "[MOVE SPLIT " << stage << "] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
        bmms_t = proposed_bayeslm;
      } 
    }
    
    if(move_type == 1){
      //clog << "A";
      add_proposed ++;
      
      propose_splitmask = split_add2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                      bmms_t.mask_nosplits,
                                      rnd_moving_lev, 
                                      to_from_ratio);
      //cout << propose_splitmask << endl;
      propose_splitsub = splitmask_to_splitsub(propose_splitmask);
      //cout << propose_splitsub.row(rnd_moving) << endl;
      ModularLR2D proposed_bayeslm = bmms_t;
      proposed_bayeslm.change_module(rnd_moving_lev, propose_splitsub);
      
      //clog << rnd_moving_lev << " rows: " << bmms_t.modules[rnd_moving_lev].splitmat.n_rows << endl;
      double totsplit_prior_mhr = totsplit_prior_ratio(propose_splitsub(rnd_moving_lev).n_rows, 
                                                       bmms_t.modules[rnd_moving_lev].splitmat.n_rows, n, 
                                                       rnd_moving_lev, 
                                                       lambda_centers);
      
      mhr = exp(proposed_bayeslm.logliks(rnd_moving_lev) - 
        bmms_t.logliks(rnd_moving_lev)) * 
        //exp(arma::accu(proposed_bayeslm.logliks.subvec(rnd_moving_lev, proposed_bayeslm.n_stages-1)) - 
        //arma::accu(bmms_t.logliks.subvec(rnd_moving_lev, bmms_t.n_stages-1))) * 
        to_from_ratio * totsplit_prior_mhr;
      mhr = mhr > 1 ? 1 : mhr;
      //cout << mhr << endl;
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        //clog << "ADDED " << endl;
        add_accepted ++;
        bmms_t = proposed_bayeslm;
      } 
      //clog << "add " << totsplit_prior_mhr << endl;
    }
    if(move_type == 2){
      //clog << "D";
      int num_splits = bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).n_rows;
      
      if(num_splits>2){
        drop_proposed ++;
        //clog << "at level " << rnd_moving_lev << endl;
        propose_splitmask = split_drop2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                         bmms_t.mask_nosplits,
                                         rnd_moving_lev,  
                                         to_from_ratio);
        
        
        propose_splitsub = splitmask_to_splitsub(propose_splitmask);
        
        ModularLR2D proposed_bayeslm = bmms_t;
        //clog << "changing " << endl;
        proposed_bayeslm.change_module(rnd_moving_lev, propose_splitsub);
        //clog << "done" << endl;
        
        //clog << "seeing this " << endl;
        double totsplit_prior_mhr = totsplit_prior_ratio(propose_splitsub(rnd_moving_lev).n_rows, 
                                                         bmms_t.modules[rnd_moving_lev].splitmat.n_rows, n, 
                                                         rnd_moving_lev,  
                                                         lambda_centers);
        
        mhr = exp(proposed_bayeslm.logliks(rnd_moving_lev) - 
          bmms_t.logliks(rnd_moving_lev)) *
          //exp(arma::accu(proposed_bayeslm.logliks.subvec(rnd_moving_lev, proposed_bayeslm.n_stages-1)) - 
          //arma::accu(bmms_t.logliks.subvec(rnd_moving_lev, bmms_t.n_stages-1))) * 
          to_from_ratio * totsplit_prior_mhr;
        mhr = mhr > 1 ? 1 : mhr;
        //clog << "and this " << endl;
        int accepted_proposal = rndpp_discrete({1-mhr, mhr});
        if(accepted_proposal == 1){
          drop_accepted ++;
          //clog << "DROPPED " << endl;
          bmms_t = proposed_bayeslm;
        }
        //clog << "drop " << totsplit_prior_mhr << endl;
        
      } else {
        //clog << "didnt" << endl;
      }
      
    }
    
    if(move_type == 99){
      splitpar_proposed++;
      //clog << "Q"; mh for split par
      double splitpar_m = arma::randn<double>();
      double new_lambda_centers = 0.1 + lambda_centers * exp(splitpar_m);
      
      mhr = exp(splitpar_prior(new_lambda_centers, bmms_t.modules[bmms_t.n_stages-1].splitmat.n_rows, n, max_stages-1) - 
        splitpar_prior(lambda_centers, bmms_t.modules[bmms_t.n_stages-1].splitmat.n_rows, n, max_stages-1)) * 
        exp(gammaprior_mhr(new_lambda_centers, lambda_centers));
      mhr = mhr > 1 ? 1 : mhr;
      
      int accepted_proposal = rndpp_discrete({1-mhr, mhr});
      if(accepted_proposal == 1){
        lambda_centers = new_lambda_centers;
        //clog << lambda_centers << endl;
        splitpar_accepted++;
      }
    }
    num_levs = bmms_t.n_stages;
    
    // gibbs latent
    z = mvtruncnormal_eye1(bmms_t.Xb_sum, trunc_lowerlim, trunc_upperlim).col(0);
    bmms_t = ModularLR2D(z, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma);
    
    
    if(m>burn-1){
      int i = m-burn;
      theta_mcmc(i) = bmms_t.theta_sampled;
      icept_mcmc(i) = bmms_t.icept_sampled;
      splitsub_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].splitmat;
      lambda_mcmc(i) = bmms_t.lambda;
      splitsparam_mcmc(i) = lambda_centers;
      dim_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].effective_dimension;
      zsave.col(i) = z;
      if(save_splitmask == true){
        splitmask_mcmc.slice(i) = bmms_t.modules[bmms_t.n_stages-1].splitmask;
      };
      Xflat_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].X_flat;
    }
    
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        Rcpp::checkUserInterrupt();
        clog << endl << 
          partnum << " " << floor(100.0*(m+0.0)/mcmc) << 
            " " << bmms_t.n_stages << " " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << 
              " " << add_accepted / (add_proposed+0.0) << " " << drop_accepted / (drop_proposed+0.0) << " " <<
                splitpar_accepted / (splitpar_proposed+0.0) << " z " << z.max() << endl;
        if(save){
          bmms_t.modules[bmms_t.n_stages-1].splitmat.save("bmms_centers.temp");
        }
        
      } 
    }
    
  }
  
  return Rcpp::List::create(
    //Rcpp::Named("mu_post") = bayeslm.mu_post,
    //Rcpp::Named("Sigma_post") = bayeslm.Sigma_post,
    //Rcpp::Named("regions") = bayeslm.regions,
    //Rcpp::Named("groupmask") = bmms_t.modules[bmms_t.n_stages-1].groupmask,
    //Rcpp::Named("splitsub") = splitsub_mcmc,
    Rcpp::Named("dimension_overall") = dim_mcmc,
    Rcpp::Named("splitsparam") = splitsparam_mcmc,
    Rcpp::Named("lambda_ridge") = lambda_mcmc,
    Rcpp::Named("flatmodeldata") = Xflat_mcmc,
    //Rcpp::Named("splitmask") = splitmask_mcmc,
    //Rcpp::Named("sigmasq_post_mean") = bmms_t.modules[bmms_t.n_stages-1].sigmasq_post_mean,
    //Rcpp::Named("a_post") = bmms_t.modules[bmms_t.n_stages-1].a_post,
    //Rcpp::Named("b_post") = bmms_t.modules[bmms_t.n_stages-1].b_post,
    //Rcpp::Named("z") = zsave,
    Rcpp::Named("icept") = icept_mcmc,
    Rcpp::Named("theta_mc") = theta_mcmc
    //Rcpp::Named("sigmasq_sampled") = bmms_t.modules[bmms_t.n_stages-1].sigmasq_sampled
  );
}

