//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#include "image_mcmc_helper.h"
using namespace std;


//' @export
//[[Rcpp::export]]
arma::field<arma::mat> load_splits(int maxlevs, std::string sname){
  arma::field<arma::mat> splits(maxlevs);
  splits.load(sname);
  //splits.print();
  return splits;
}

//[[Rcpp::export]]
arma::mat square_coord(const arma::vec& onesplit, int p1, int p2, int radius_int){
  int x1 = onesplit(0) - radius_int;
  int x2 = onesplit(0) + radius_int;
  int y1 = onesplit(1) - radius_int;
  int y2 = onesplit(1) + radius_int;
  int xlim = p1-1;
  int ylim = p2-1;
  
  arma::mat coorded = arma::zeros(2,2);
  coorded(0,0) = max(0, x1); //x1
  coorded(1,0) = min(x2, xlim); 
  coorded(0,1) = max(0, y1); //y1
  coorded(1,1) = min(y2, ylim);
  return coorded;
}


// given a mask of splits,
// move onesplit to another location around it (radius fixed)
// return new mask of splits
// also gives ratio: [number of moves TO proposed]/[number of moves FROM proposed]
// = [prob moving FROM proposed back]/[prob moving TO proposed forward]
//[[Rcpp::export]]
arma::mat split_move2d(const arma::mat& mask_of_splits, 
                       const arma::mat& mask_nosplits,
                       const arma::vec& onesplit, 
                       double& to_from_ratio, int radius_int=1){
  // coordinates to focus on
  
  //  (4,5) center, 
  arma::mat sq_coord = square_coord(onesplit, mask_of_splits.n_rows, mask_of_splits.n_cols, radius_int);
  int x1 = sq_coord(0,0);
  int x2 = sq_coord(1,0);
  int y1 = sq_coord(0,1);
  int y2 = sq_coord(1,1);
  
  // focusing
  arma::mat around_split = mask_of_splits.submat(x1, y1, x2, y2);
  arma::mat around_nosplit = mask_nosplits.submat(x1, y1, x2, y2);
  //cout << "focusing " << endl << around_split << endl;
  // pick random location among available
  // any location that is a center at any level will have >1
  arma::uvec available_locations = arma::find((around_split+around_nosplit)==1);
  
  // count moves TO
  double avail_to = arma::accu(1-around_split.elem(available_locations));
  //cout << "number of moves TO new: " << avail_to << endl;
  
  // choose random location among available ones
  // in focused view
  //rand_location = available_locations.elem(rand_location);
  int rloc = arma::randi<int>(arma::distr_param(0, available_locations.n_elem-1));
  arma::uword rand_location = rloc;
  rand_location = available_locations(rand_location);
  
  // original location in focused coordinate system
  arma::vec orig_location = arma::zeros(2);
  orig_location(0) = onesplit(0) - x1;
  orig_location(1) = onesplit(1) - y1;
  
  // change values in the focused view
  // IF we're moving to an occupied location, switch the values.
  //int old_val = arma::conv_to<int>::from(around_split.elem(rand_location));
  around_split(arma::ind2sub(arma::size(around_split), rand_location)).fill(mask_of_splits(onesplit(0), onesplit(1)));
  around_split(orig_location(0), orig_location(1)) = 0;//old_val;
  
  // substitute on original matrix
  arma::mat returning_mat = mask_of_splits;
  //returning_mat.submat(x1, y1, x2, y2) = around_split;
  
  // the moved split has these coordinates
  arma::mat new_location = arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(around_split), rand_location));
  // new coordinates in previous reference plane
  new_location(0) = new_location(0) + x1;
  new_location(1) = new_location(1) + y1;
  //cout << "new location in original coordinates : " << new_location.t() << endl;
  
  returning_mat(new_location(0), new_location(1)) = mask_of_splits(onesplit(0), onesplit(1));
  returning_mat(onesplit(0), onesplit(1)) = 0;
  /*
   arma::mat avail_around_moved = bm2d::splitmask_focus(returning_mat, new_location, radius_int);
   arma::mat avail_around_nosplit = bm2d::splitmask_focus(returning_mat, new_location, radius_int);
   double avail_from = arma::accu(1-avail_around_moved.elem(arma::find((avail_around_moved+avail_around_nosplit) == 1)));
   //number_availables(avail_around_moved);
   //cout << "number of moves FROM new: " << avail_from << endl;
   
   to_from_ratio = avail_to/avail_from;*/
  return returning_mat;
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
  int rloc = arma::randi<int>(arma::distr_param(0, available_locations.n_elem-1));
  arma::uword rand_location = rloc; //bmrandom::sample_index(1, available_locations.n_elem);
  rand_location = available_locations(rand_location);
  
  // change values in the focused view
  mask_of_splits(rand_location) = (lev+1);
  
  // occupied locations
  available_locations = arma::find(mask_of_splits>0);
  double avail_from = available_locations.n_elem;
  //cout << "number of moves FROM new: " << avail_from << endl;
  
  // ways to add 1 split: #availables
  // ways to reverse that: #occupied
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
  int rloc = arma::randi<int>(arma::distr_param(0, available_locations.n_elem-1));
  arma::uword rand_location = rloc;
  //arma::uvec rand_location = bmrandom::sample_index(1, available_locations.n_elem);
  rand_location = available_locations(rand_location);
  
  // change values in the focused view
  arma::uvec coord = arma::ind2sub(arma::size(mask_nosplits), rand_location);
  //clog << coord << endl;
  mask_of_splits(coord(0), coord(1)) = 0.0;
  //clog << mask_temp << endl;
  
  // unoccupied locations
  available_locations = arma::find((mask_of_splits + mask_nosplits)==1);
  double avail_from = arma::accu(1-mask_of_splits.elem(available_locations));
  //cout << "number of moves FROM new: " << avail_from << endl;
  
  // ways to remove
  to_from_ratio = avail_to/avail_from;
  //mask_temp.elem(arma::find(mask_nosplits==0)).fill(0);
  return mask_of_splits;
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


//[[Rcpp::export]]
arma::field<arma::mat> insert_empty_levels(const arma::field<arma::mat>& splitsub, 
                                           const arma::vec& nctr_at_lev){
  // dropping last split at last level results in missing dimension
  // propose_splitsub is 1 shorter than needed.
  arma::field<arma::mat> aug_splitsub(nctr_at_lev.n_elem);
  
  int next_active=0;
  for(unsigned int s=0; s<nctr_at_lev.n_elem; s++){
    if(nctr_at_lev(s) == 0){
      aug_splitsub(s) = arma::zeros(arma::size(0,2));
    } else {
      aug_splitsub(s) = splitsub(next_active);
      next_active++;
    }
  }
  return aug_splitsub;
}

arma::vec active_levs(const arma::field<arma::mat>& splitsub){
  arma::uvec actives = arma::zeros<arma::uvec>(splitsub.n_elem);
  for(unsigned int i=0; i<splitsub.n_elem; i++){
    if(splitsub(i).n_rows > 0){
      actives(i) = 1;
    }
  }
  arma::vec all_levs = arma::regspace(0, splitsub.n_elem-1);
  return all_levs.elem(arma::find(actives));
}

//'@export
//[[Rcpp::export]]
Rcpp::List soi_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> centers,
                   arma::mat mask_forbid,
                   double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius=2,
                   int start_movinglev=0, int partnum=0, 
                   bool save=false, bool save_more_data = false,
                   bool fixsigma = false,
                   arma::vec gin = arma::vec(),
                   bool try_bubbles = false){
  //former name: model_test
  
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
  
  arma::vec g = arma::zeros(max_stages)-1;
  if(gin.n_elem != 0){
    if(gin.n_elem == 1){
      g = arma::ones(max_stages) * gin;
    } else {
      g = gin.subvec(0, max_stages-1);
    }
  }
  
  arma::vec unique_y = arma::unique(y);
  bool binary = unique_y.n_elem == 2;
  arma::vec z(n);
  
  int move_proposed = 0;
  int move_accepted = 0;
  int add_proposed = 0;
  int add_accepted = 0;
  int drop_proposed = 0;
  int drop_accepted = 0;
  
  // proposal types for centers: move/add/remove
  int choices = 3;
  arma::field<arma::mat> proposal(choices);
  arma::field<arma::mat> acceptance(choices);
  for(int j=0; j<choices; j++){
    proposal(j) = arma::zeros(max_stages, mcmc);
    acceptance(j) = arma::zeros(max_stages, mcmc);
  }
  
  int p1 = X.n_rows;
  int p2 = X.n_cols;
  
  double mindim = .0 + (p1>p2? p2 : p1);
  double maxdim = .0 + (p1>p2? p1 : p2);
  double maxradius = mindim*.5/(max_stages+.0);
  double bubbles_radius = try_bubbles ? maxradius/5.0 : -1;
  
  int splitpar_proposed = 0;
  int splitpar_accepted = 0;
  
  double sigmasq_sample = fixsigma? (binary? 1.0 : arma::var(y)) :-1.0;

  arma::vec lambda_mcmc;
  arma::cube splitmask_mcmc;
  arma::field<arma::field<arma::mat>> splitsub_mcmc;
  arma::field<arma::field<arma::mat>> proposed_splitsub_mcmc;
  
  arma::mat sigmasq_mcmc(mcmc-burn, max_stages);
  arma::field<arma::cube> theta_mcmc(mcmc-burn);
  arma::field<arma::vec> icept_mcmc(mcmc-burn);
  
  arma::vec dim_mcmc(mcmc-burn);
  arma::vec bubbles_mcmc(mcmc-burn);
  arma::vec mhr_mcmc(mcmc-burn);
  
  if(save_more_data == true){
    splitsub_mcmc = arma::field<arma::field<arma::mat>>(mcmc-burn);
    proposed_splitsub_mcmc = arma::field<arma::field<arma::mat>>(mcmc-burn);
    lambda_mcmc = arma::vec(mcmc-burn);
    //splitsparam_mcmc(i) = lambda_centers;
    
    //zsave.col(i) = z;
    splitmask_mcmc = arma::zeros(X.n_rows, X.n_cols, mcmc-burn);
  };
  
  int rnd_moving_split = -1;
  int rnd_moving_lev = -1;
  
  arma::vec moved_split = arma::zeros(mcmc-burn);
  arma::vec moved_lev = arma::zeros(mcmc-burn);
  arma::vec adddrop_split = arma::zeros(mcmc-burn);
  arma::mat movdir = arma::zeros(mcmc-burn, 2);
  // initialize
  arma::mat propose_splitmask;
  arma::field<arma::mat> propose_splitsub;
  
  clog << "Initalizing - ";
  if(binary){
    clog << "binomial." << endl;
  } else {
    clog << "gaussian." << endl;
  }
  
  ModularLR2D bmms_t = ModularLR2D(y, X, centers, mask_forbid, max_stages, lambda_ridge, 
                                   fixsigma, binary,
                                   sigmasq_sample, g, bubbles_radius);

  int num_levs = bmms_t.n_stages;
  if(binary){
    z = bmms_t.modules[0].X_flat * bmms_t.modules[0].flatmodel.b;
  } else {
    z = y;
  }
  
  //clog << "Fixing sigmasq at all modules. Deliberate?" << endl;
  int m_last=0;
  for(unsigned int m = 0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    if(m==0){ 
      clog << "> starting mcmc." << endl;
    }
    
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    int accepted_proposal = 0;
    

    rnd_moving_lev = arma::randi<int>(arma::distr_param(start_movinglev, num_levs-1));

    try {
      if(move_type == 0){
        //clog << "M";
        move_proposed++;
        proposal(move_type)(rnd_moving_lev, m) = 1;
        
        int num_splits = bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).n_rows;
        
        rnd_moving_split = arma::randi<int>(arma::distr_param(0, num_splits-1));
        propose_splitmask = split_move2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                         bmms_t.mask_nosplits,
                                         bmms_t.modules[bmms_t.n_stages-1].splitmat(rnd_moving_lev).row(rnd_moving_split).t(), 
                                         to_from_ratio, radius);
        propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
        
        ModularLR2D proposed_bayeslm = bmms_t;
        proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
        
        //clog << sigmasq_sample << " " << fixsigma << endl;
        //ModularLR2D proposed_bayeslm(y, X, propose_splitsub, mask_forbid, 
         //                            max_stages, lambda_ridge, fixsigma, binary, sigmasq_sample, g, bubbles_radius);
        
        double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].flatmodel.p+1,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                          bmms_t.modules[rnd_moving_lev].flatmodel.p+1, n, 
                                                          propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                          lambda_centers);
        
        mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks)) * totsplit_prior_mhr;
        //clog << arma::join_horiz(proposed_bayeslm.logliks, bmms_t.logliks) << endl;
        //clog << totsplit_prior_mhr << ": " << proposed_bayeslm.modules[rnd_moving_lev].flatmodel.p - bmms_t.modules[rnd_moving_lev].flatmodel.p << endl;
        
        mhr = mhr > 1 ? 1 : mhr;
        accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
        if(accepted_proposal == 1){
          proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
          bmms_t = proposed_bayeslm;
          move_accepted++;
          acceptance(move_type)(rnd_moving_lev, m) = 1;
        } 
      }
      if(move_type == 1){
        //clog << "A";
        add_proposed ++;
        proposal(move_type)(rnd_moving_lev, m) = 1;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits<100){
          propose_splitmask = split_add2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                          bmms_t.mask_nosplits,
                                          rnd_moving_lev, 
                                          to_from_ratio);
          //cout << propose_splitmask << endl;
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);

          //ModularLR2D proposed_bayeslm(y, X, propose_splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma, sigmasq_sample, g, bubbles_radius);
          ModularLR2D proposed_bayeslm = bmms_t;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].flatmodel.p+1,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].flatmodel.p+1, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks)) * 
            to_from_ratio * totsplit_prior_mhr;
          
          mhr = mhr > 1 ? 1 : mhr;
          //clog << arma::join_horiz(proposed_bayeslm.logliks, bmms_t.logliks) << endl;
          //clog << to_from_ratio << " " << totsplit_prior_mhr << endl;
          
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            //clog << "ADDED " << endl;
            add_accepted ++;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
            acceptance(move_type)(rnd_moving_lev, m) = 1;
          } 
          //clog << "add " << totsplit_prior_mhr << endl;
        }
      }
      if(move_type == 2){
        //clog << "D";
        drop_proposed ++;
        proposal(move_type)(rnd_moving_lev, m) = 1;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        int min_num_splits = try_bubbles ? 1 : 1;
        
        if(num_splits > min_num_splits){
          propose_splitmask = split_drop2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                           bmms_t.mask_nosplits,
                                           rnd_moving_lev,  
                                           to_from_ratio);

          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);

          //ModularLR2D proposed_bayeslm(y, X, propose_splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma, sigmasq_sample, g, bubbles_radius);
          ModularLR2D proposed_bayeslm = bmms_t;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].flatmodel.p+1,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].flatmodel.p+1, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks-bmms_t.logliks)) * 
            to_from_ratio * totsplit_prior_mhr;
          //clog << "drop: " << to_from_ratio << " " << totsplit_prior_mhr << endl;
          mhr = mhr > 1 ? 1 : mhr;
          //clog << "and this " << endl;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            drop_accepted ++;
            //clog << "DROPPED " << proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << " from " << bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << endl;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
            acceptance(move_type)(rnd_moving_lev, m) = 1;
          }
        }
      }
    } catch(...) {
      //clog << bmms_t.splitsub << endl;
      //clog << "--" << endl;
      //clog << propose_splitsub << endl;
      //clog << rnd_moving_lev << endl;
      clog << "Error. Failed move. Type: (" << move_type << ")" << endl;
      break;
    }
    
    try {
      if(!binary){
        if(fixsigma){
          BayesLM2D lastmodule = bmms_t.modules[bmms_t.n_stages-1];
          //cout << "[MCMC sampling sigmasq] detPx " << arma::det(Px) << endl;
          double beta_post = arma::conv_to<double>::from(.5*(lastmodule.y.t() * (lastmodule.In - lastmodule.flatmodel.Px) * lastmodule.y));
          //cout << "[MCMC sampling sigmasq] beta_post " << beta_post << endl;
          sigmasq_sample = 1.0 / bmrandom::rndpp_gamma(0.1 + n/2.0, 1.0/(0.1 + beta_post));
          bmms_t = ModularLR2D(y, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, 
                                  fixsigma, binary,
                                  sigmasq_sample, g, bubbles_radius);
        } else {
          sigmasq_sample = -1.0;
        }
      } else {
        arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), bmms_t.Xb_sum);
        z = 1.0/w % (y-.5);
        bmms_t = ModularLR2D(z, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, 
                             fixsigma, binary, -1.0, g, bubbles_radius);
      }
    } catch(...){
      clog << "Error. Failed sampling random component." << endl;
      break;
    }
    
    if(try_bubbles){
      double radius_propose=1.0;
      try {
        //try{
          double rmove = R::rnorm(0,1);
          radius_propose = exp( log(bubbles_radius) + rmove*.5 );
          if(abs(radius_propose) > maxradius){
            radius_propose = maxradius;
          }
          if(abs(radius_propose) < 1){
            radius_propose = 1.0;
          }
          if(radius_propose != bubbles_radius){
            ModularLR2D bmms_t_prop(y, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, 
                                    fixsigma, binary,
                                    sigmasq_sample, g, radius_propose);
            if(!bmms_t_prop.logliks.has_inf()){
              double totsplit_prior_mhr = totsplit_prior2_ratio(bmms_t_prop.modules[bmms_t_prop.n_stages-1].flatmodel.p+1,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                                bmms_t.modules[bmms_t.n_stages-1].flatmodel.p+1, n, 
                                                                bmms_t.n_stages - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                                lambda_centers);
              
              double prob = exp(arma::accu(bmms_t_prop.logliks - bmms_t.logliks)) * totsplit_prior_mhr *
                exp( 5/maxradius*(- radius_propose + bubbles_radius)) * radius_propose / bubbles_radius; //exp(1) prior
              
              
              prob = prob > 1 ? 1 : prob;
              accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
              if(accepted_proposal == 1){
                bubbles_radius = radius_propose;
                bmms_t = bmms_t_prop;
              } 
            }
          }
      } catch(...){
        clog << m << " Failed sampling bubbles radius. Skip. " << radius_propose << endl;
        bubbles_radius = m>burn? bubbles_mcmc(m-burn) : maxradius/5.0;
      }
    } 
      if(m>burn-1){
        int i = m-burn;
        theta_mcmc(i) = bmms_t.theta_sampled;
        icept_mcmc(i) = bmms_t.icept_sampled;
        dim_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].effective_dimension;
        bubbles_mcmc(i) = bubbles_radius;
        
        if(save_more_data == true){
          splitsub_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].splitmat;
          proposed_splitsub_mcmc(i) = propose_splitsub;
          moved_split(i) = rnd_moving_split;
          moved_lev(i) = rnd_moving_lev;
          mhr_mcmc(i) = mhr;
          for(int s=0; s<max_stages; s++){
            sigmasq_mcmc(i,s) = bmms_t.modules[s].sigmasq_sampled;
          }
          
        }
      }
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        Rcpp::checkUserInterrupt();
        clog << endl << 
          partnum << " [" << floor(100.0*(m+0.0)/mcmc) << 
            "%] " << bmms_t.n_stages << " " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << 
              " a:" << add_accepted / (add_proposed+0.0) << " d:" << drop_accepted / (drop_proposed+0.0) << " m:" <<
                move_accepted / (move_proposed+0.0) << ". z " << z.max() << endl;
        if(save){
          bmms_t.modules[bmms_t.n_stages-1].splitmat.save("bmms_centers.temp");
        }
      } 
    }
    m_last++;
  } // mcmc
  if(save_more_data){
    return Rcpp::List::create(
      Rcpp::Named("m") = m_last,
      Rcpp::Named("rnd_moving_lev") = rnd_moving_lev,
      Rcpp::Named("mhr_mcmc") = mhr_mcmc,
      Rcpp::Named("splitsub") = splitsub_mcmc,
      Rcpp::Named("proposed_splitsub") = proposed_splitsub_mcmc,
      Rcpp::Named("dimension_overall") = dim_mcmc,
      Rcpp::Named("bubbles") = bubbles_mcmc,
      Rcpp::Named("acceptance") = acceptance,
      Rcpp::Named("proposal") = proposal,
      //Rcpp::Named("mask_splits") = bmms_t.modules[bmms_t.n_stages-1].splitmask,
      //Rcpp::Named("mask_nosplits") = bmms_t.mask_nosplits,
      Rcpp::Named("icept") = icept_mcmc,
      Rcpp::Named("theta_mc") = theta_mcmc,
      Rcpp::Named("sigmasq") = sigmasq_mcmc
    );
  } else {
    return Rcpp::List::create(
      Rcpp::Named("m") = m_last,
      Rcpp::Named("dimension_overall") = dim_mcmc,
      Rcpp::Named("icept") = icept_mcmc,
      Rcpp::Named("theta_mc") = theta_mcmc,
      Rcpp::Named("bubbles") = bubbles_mcmc,
      Rcpp::Named("sigmasq") = sigmasq_mcmc
    );
  }
}


//'@export
//[[Rcpp::export]]
Rcpp::List soi_tester(arma::vec y, arma::cube X, 
                      arma::field<arma::mat> centers,
                      arma::field<arma::mat> to_centers,
                      int levelchg,
                      arma::mat mask_forbid,
                      double sigmasq,
                      double lambda_ridge, 
                      bool fixsigma = false,
                      arma::vec gin = arma::vec(),
                      double bubbles_radius = -1.0){
  //former name: model_test
  
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
  arma::vec yunique = arma::unique(y);
  bool binary = yunique.n_elem == 2;
  
  arma::vec g = arma::zeros(max_stages)-1;
  if(gin.n_elem != 0){
    g = gin.subvec(0, max_stages-1);
  }
  
  clog << "initalize " << endl;
  ModularLR2D bmms_t = ModularLR2D(y, X, centers, mask_forbid, max_stages, lambda_ridge, 
                                   fixsigma, binary,
                                   sigmasq, g, bubbles_radius);
  
  arma::vec sigmasq_sampled = bmms_t.sigmasq_sampled;
  arma::vec loglik_before = bmms_t.logliks;
  bmms_t.propose_change_module(levelchg, to_centers);
  
  return Rcpp::List::create(
    Rcpp::Named("logpost_before") = loglik_before,
    Rcpp::Named("logpost_after") = bmms_t.logliks,
    Rcpp::Named("g") = bmms_t.g,
    Rcpp::Named("lambda_ridge") = bmms_t.lambda,
    Rcpp::Named("logpost") = bmms_t.logliks,
    Rcpp::Named("splitsub") = bmms_t.splitsub,
    Rcpp::Named("sigmasq_sampled") = sigmasq_sampled,
    Rcpp::Named("sigmasq_sampled_after") = bmms_t.sigmasq_sampled,
    Rcpp::Named("theta") = bmms_t.theta_sampled,
    Rcpp::Named("icept") = bmms_t.icept_sampled,
    Rcpp::Named("Xb_sum") = bmms_t.Xb_sum
    
  );
}

/*
//'@export
//[[Rcpp::export]]
Rcpp::List soi_binary_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> centers,
                          arma::mat mask_forbid,
                          double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius=2,
                          int start_movinglev=0,
                          int partnum=0, bool save=true,
                          bool save_more_data = true,
                          bool fixsigma = false,
                          double g = -1.0,
                          bool try_bubbles =false){
  //former name: model_test
  
  // vectorize X using splits
  // build blr model
  // propose split changes [move]
  // eval mhr & accept/reject
  // reconstruct & save beta at this step of the chain
  // start_movinglev = to fix hemispheres as first level
  int n = y.n_elem;
  int p1 = X.n_rows;
  int p2 = X.n_cols;
  double bubbles_radius = try_bubbles? (p1+p2+.0)*.5 : -1.0;
  
  double to_from_ratio = 0.0;
  //int radius = 2;
  double mhr = 0.0;
  int max_stages = centers.n_elem;
  
  int move_proposed = 0;
  int move_accepted = 0;
  
  int add_proposed = 0;
  int add_accepted = 0;
  int drop_proposed = 0;
  int drop_accepted = 0;
  
  int splitpar_proposed = 0;
  int splitpar_accepted = 0;
  
  //splits = merge_splits(splits, old_splits);
  
  //clog << splits << endl;
  //arma::vec splitsparam_mcmc(mcmc-burn);
  arma::vec lambda_mcmc;
  arma::cube splitmask_mcmc;
  arma::field<arma::field<arma::mat>> splitsub_mcmc;
  
  arma::field<arma::cube> theta_mcmc(mcmc-burn);
  arma::field<arma::vec> icept_mcmc(mcmc-burn);
  arma::vec dim_mcmc(mcmc-burn);
  
  
  if(save_more_data == true){
    splitsub_mcmc = arma::field<arma::field<arma::mat>>(mcmc-burn);
    lambda_mcmc = arma::vec(mcmc-burn);
    //splitsparam_mcmc(i) = lambda_centers;
    
    //zsave.col(i) = z;
    splitmask_mcmc = arma::zeros(X.n_rows, X.n_cols, mcmc-burn);
  };
  
  //arma::field<arma::mat> Xflat_mcmc(mcmc-burn);
  arma::vec ybin(n);
  arma::vec z(n);
  arma::mat zsave(n, mcmc-burn);
  
  // initialize
  arma::mat propose_splitmask;
  arma::field<arma::mat> propose_splitsub;
  
  clog << "initalize " << endl;
  ModularLR2D bmms_t = ModularLR2D(y, X, centers, mask_forbid, max_stages, lambda_ridge, fixsigma, 1.0, g, bubbles_radius);
  int num_levs = bmms_t.n_stages;
  
  ybin = y;
  z = bmms_t.modules[0].X_flat * bmms_t.modules[0].flatmodel.b;
  
  for(unsigned int m = 0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    if(m==0){ 
      clog << "starting mcmc" << endl;
    }
    int choices = 3;
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    int rnd_moving_lev = arma::randi<int>(arma::distr_param(start_movinglev, num_levs-1));
    int accepted_proposal = 0;
    try {
      if(move_type == 0){
        //clog << "M";
        move_proposed++;
        
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
        
        propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
        
        ModularLR2D proposed_bayeslm = bmms_t;
        
        //clog << "changing " << endl;
        proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
        //clog << "done" << endl;
        //clog << exp(arma::accu(proposed_bayeslm.logliks-bmms_t.logliks)) << endl;
        mhr = exp(arma::accu(proposed_bayeslm.logliks-bmms_t.logliks));
        
        mhr = mhr > 1 ? 1 : mhr;
        
        //clog << "moving mhr " << mhr << endl << "to_from_ratio " << to_from_ratio << endl;
        
        accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
        if(accepted_proposal == 1){
          //clog << "[MOVE SPLIT " << stage << "] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
          proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
          bmms_t = proposed_bayeslm;
          move_accepted++;
        } 
      }
      if(move_type == 1){
        //clog << "A";
        add_proposed ++;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits<100){
          propose_splitmask = split_add2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                          bmms_t.mask_nosplits,
                                          rnd_moving_lev, 
                                          to_from_ratio);
          //cout << propose_splitmask << endl;
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
          //cout << propose_splitsub.row(rnd_moving) << endl;
          ModularLR2D proposed_bayeslm = bmms_t;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks)) * to_from_ratio * totsplit_prior_mhr;
          
          mhr = mhr > 1 ? 1 : mhr;
          //cout << mhr << endl;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            //clog << "ADDED " << endl;
            add_accepted ++;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
          } 
          //clog << "add " << totsplit_prior_mhr << endl;
        }
      }
      if(move_type == 2){
        //clog << "D";
        drop_proposed ++;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits>2){
          propose_splitmask = split_drop2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                           bmms_t.mask_nosplits,
                                           rnd_moving_lev,  
                                           to_from_ratio);
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
          
          ModularLR2D proposed_bayeslm = bmms_t;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          //clog << "done" << endl;
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks-bmms_t.logliks)) * to_from_ratio * totsplit_prior_mhr;
          //clog << "drop: " << to_from_ratio << " " << totsplit_prior_mhr << endl;
          mhr = mhr > 1 ? 1 : mhr;
          //clog << "and this " << endl;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            drop_accepted ++;
            //clog << "DROPPED " << proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << " from " << bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << endl;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
          }
        }
      }
    } catch(...) {
      clog << "Skipping failed move. Type: (" << move_type << ")" << endl;
    }
    arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), bmms_t.Xb_sum);
    z = 1.0/w % (y-.5);
    bmms_t = ModularLR2D(z, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma, 1.0, g, bubbles_radius);
    
    if(m>burn-1){
      int i = m-burn;
      theta_mcmc(i) = bmms_t.theta_sampled;
      icept_mcmc(i) = bmms_t.icept_sampled;
      dim_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].effective_dimension;
      if(save_more_data == true){
        splitsub_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].splitmat;
        splitmask_mcmc.slice(i) = bmms_t.modules[bmms_t.n_stages-1].splitmask;
      };
    }
    
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        Rcpp::checkUserInterrupt();
        clog << endl << 
          partnum << " " << floor(100.0*(m+0.0)/mcmc) << 
            " " << bmms_t.n_stages << " " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << 
              " a:" << add_accepted / (add_proposed+0.0) << " d:" << drop_accepted / (drop_proposed+0.0) << " m:" <<
                move_accepted / (move_proposed+0.0) << " (s:" <<
                  splitpar_accepted / (splitpar_proposed+0.0) << ") z " << z.max() << endl;
        if(save){
          bmms_t.modules[bmms_t.n_stages-1].splitmat.save("bmms_centers.temp");
        }
      } 
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("dimension_overall") = dim_mcmc,
    Rcpp::Named("icept") = icept_mcmc,
    Rcpp::Named("theta_mc") = theta_mcmc
  );
}
*/

//'@export
//[[Rcpp::export]]
Rcpp::List mixed_binary_cpp(arma::vec y, arma::cube X, arma::mat X_g, 
                            arma::field<arma::mat> centers,
                            arma::mat mask_forbid,
                            double lambda_centers, double lambda_ridge, 
                            int mcmc, int burn, int radius=2,
                            int start_movinglev=0, int partnum=0, 
                            bool save=true, bool save_more_data = true,
                            bool fixsigma = false,
                            double g = -1.0,
                            double g_vs = 1.0,
                            double module_prior_par_vs=1.0){
  
  int n = y.n_elem;
  double to_from_ratio = 0.0;
  double mhr = 0.0;
  int max_stages = centers.n_elem;
  
  int move_proposed = 0;
  int move_accepted = 0;
  
  int add_proposed = 0;
  int add_accepted = 0;
  int drop_proposed = 0;
  int drop_accepted = 0;
  
  int splitpar_proposed = 0;
  int splitpar_accepted = 0;
  
  arma::vec lambda_mcmc;
  arma::cube splitmask_mcmc;
  arma::field<arma::field<arma::mat>> splitsub_mcmc;
  
  arma::field<arma::cube> theta_mcmc(mcmc-burn);
  arma::field<arma::vec> icept_mcmc(mcmc-burn);
  arma::vec dim_mcmc(mcmc-burn);
  
  if(save_more_data == true){
    splitsub_mcmc = arma::field<arma::field<arma::mat>>(mcmc-burn);
    lambda_mcmc = arma::vec(mcmc-burn);
    splitmask_mcmc = arma::zeros(X.n_rows, X.n_cols, mcmc-burn);
  };
  
  arma::vec z(n);
  arma::mat zsave(n, mcmc-burn);
  
  // initialize
  arma::mat propose_splitmask;
  arma::field<arma::mat> propose_splitsub;
  
  clog << "initalize " << endl;
  ModularLR2D bmms_t = ModularLR2D(y, X, centers, mask_forbid, max_stages, lambda_ridge, fixsigma, 1.0, g);
  int num_levs = bmms_t.n_stages;
  
  z = bmms_t.modules[0].X_flat * bmms_t.modules[0].flatmodel.b;
  
  // vs last module
  arma::vec gamma_start = arma::zeros(X_g.n_cols)+.1;
  arma::vec g_intercept = arma::zeros(mcmc-burn);
  arma::mat g_beta_store = arma::zeros(X_g.n_cols, mcmc-burn);
  arma::mat g_gamma_store = arma::zeros(X_g.n_cols, mcmc-burn);
  double gg_g = g_vs;
  double module_prior_par = module_prior_par_vs;
  bmmodels::VarSelMCMC vsmodule(z, X_g, gamma_start, gg_g, module_prior_par, false, 1); // true = binary
  
  for(unsigned int m = 0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    if(m==0){ 
      clog << "starting mcmc" << endl;
    }
    int choices = 3;
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    int rnd_moving_lev = arma::randi<int>(arma::distr_param(start_movinglev, num_levs-1));
    int accepted_proposal = 0;
    try {
      
      if(move_type == 0){
        //clog << "M";
        move_proposed++;
        
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
        
        propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
        
        ModularLR2D proposed_bayeslm = bmms_t;
        
        //clog << "changing " << endl;
        proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
        //clog << "done" << endl;
        //clog << exp(arma::accu(proposed_bayeslm.logliks-bmms_t.logliks)) << endl;
        arma::vec resid = bmms_t.modules[bmms_t.n_stages-1].residuals;
        bmmodels::VarSelMCMC propose_vsmodule(resid, X_g, gamma_start, gg_g, module_prior_par, false, 0); // true = binary
        
        mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks) + 
          propose_vsmodule.marglik - vsmodule.marglik);
        
        mhr = mhr > 1 ? 1 : mhr;
        
        //clog << "moving mhr " << mhr << endl << "to_from_ratio " << to_from_ratio << endl;
        
        accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
        if(accepted_proposal == 1){
          //clog << "[MOVE SPLIT " << stage << "] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
          proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
          bmms_t = proposed_bayeslm;
          move_accepted++;
        } 
      }
      if(move_type == 1){
        //clog << "A";
        add_proposed ++;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits<100){
          propose_splitmask = split_add2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                          bmms_t.mask_nosplits,
                                          rnd_moving_lev, 
                                          to_from_ratio);
          //cout << propose_splitmask << endl;
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
          //cout << propose_splitsub.row(rnd_moving) << endl;
          ModularLR2D proposed_bayeslm = bmms_t;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          //mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks)) * to_from_ratio * totsplit_prior_mhr;
          
          arma::vec resid = bmms_t.modules[bmms_t.n_stages-1].residuals;
          bmmodels::VarSelMCMC propose_vsmodule(resid, X_g, gamma_start, gg_g, module_prior_par, false, 0); // true = binary
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks) + 
            propose_vsmodule.marglik - vsmodule.marglik) * to_from_ratio * totsplit_prior_mhr;
          
          mhr = mhr > 1 ? 1 : mhr;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            //clog << "ADDED " << endl;
            add_accepted ++;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
          } 
          //clog << "add " << totsplit_prior_mhr << endl;
        }
      }
      if(move_type == 2){
        //clog << "D";
        drop_proposed ++;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits>2){
          
          
          propose_splitmask = split_drop2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                           bmms_t.mask_nosplits,
                                           rnd_moving_lev,  
                                           to_from_ratio);
          
          
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
          
          ModularLR2D proposed_bayeslm = bmms_t;
          //clog << "changing " << endl;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          //clog << "done" << endl;
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          
          arma::vec resid = bmms_t.modules[bmms_t.n_stages-1].residuals;
          bmmodels::VarSelMCMC propose_vsmodule(resid, X_g, gamma_start, gg_g, module_prior_par, false, 0); // true = binary
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks) + 
            propose_vsmodule.marglik - vsmodule.marglik) * to_from_ratio * totsplit_prior_mhr;
          
          //clog << "drop: " << to_from_ratio << " " << totsplit_prior_mhr << endl;
          mhr = mhr > 1 ? 1 : mhr;
          //clog << "and this " << endl;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            drop_accepted ++;
            //clog << "DROPPED " << proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << " from " << bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << endl;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
          }
          //clog << "drop " << totsplit_prior_mhr << endl;
          
        } else {
          //clog << "didnt" << endl;
        }
        
      }
      
      num_levs = bmms_t.n_stages;
      
    } catch(...) {
      clog << "Skipping failed move. Type: (" << move_type << ")" << endl;
    }
    
    
    //////////////// variable selection on finest scale (Gordon 333)
    arma::vec resid = bmms_t.modules[bmms_t.n_stages-1].residuals;
    
    vsmodule = bmmodels::VarSelMCMC(resid, X_g, gamma_start, gg_g, module_prior_par, false, 1); // true = binary
    //clog << "SEL: #" << arma::sum(vsmodule.gamma_stored.col(0)) << endl;
    arma::mat g_xb = X_g * vsmodule.beta_stored.col(0) + vsmodule.icept_stored(0);
    //clog << "  gamma start" << endl;
    gamma_start = vsmodule.gamma_stored.col(0);
    //clog << gamma_start(1) << endl;
    
    // gibbs latent
    //z = bmrandom::mvtruncnormal_eye1(bmms_t.Xb_sum, trunc_lowerlim, trunc_upperlim).col(0);
    arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), bmms_t.Xb_sum + g_xb);
    z = 1.0/w % (y-.5);
    bmms_t = ModularLR2D(z, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma, 1.0, g);
    
    if(m>burn-1){
      int i = m-burn;
      theta_mcmc(i) = bmms_t.theta_sampled;
      icept_mcmc(i) = bmms_t.icept_sampled;
      dim_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].effective_dimension;
      g_beta_store.col(i) = vsmodule.beta_stored.col(0);
      g_gamma_store.col(i) = vsmodule.gamma_stored.col(0);
      g_intercept(i) = vsmodule.icept_stored(0);// vsmodule.intercept;
      if(save_more_data == true){
        splitsub_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].splitmat;
        //lambda_mcmc(i) = bmms_t.lambda;
        //splitsparam_mcmc(i) = lambda_centers;
        //zsave.col(i) = z;
        splitmask_mcmc.slice(i) = bmms_t.modules[bmms_t.n_stages-1].splitmask;
      };
      //Xflat_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].X_flat;
    }
    
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        Rcpp::checkUserInterrupt();
        clog << endl << 
          partnum << " " << floor(100.0*(m+0.0)/mcmc) << 
            " " << bmms_t.n_stages << " " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << 
              " a:" << add_accepted / (add_proposed+0.0) << " d:" << drop_accepted / (drop_proposed+0.0) << " m:" <<
                move_accepted / (move_proposed+0.0) << " #(" << arma::sum(vsmodule.gamma_stored.col(0)) << 
                  ") z " << z.max() << endl;
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
    //Rcpp::Named("splitsparam") = splitsparam_mcmc,
    //Rcpp::Named("lambda_ridge") = lambda_mcmc,
    //Rcpp::Named("splitmask") = splitmask_mcmc,
    //Rcpp::Named("sigmasq_post_mean") = bmms_t.modules[bmms_t.n_stages-1].sigmasq_post_mean,
    //Rcpp::Named("a_post") = bmms_t.modules[bmms_t.n_stages-1].a_post,
    //Rcpp::Named("b_post") = bmms_t.modules[bmms_t.n_stages-1].b_post,
    //Rcpp::Named("z") = zsave,
    Rcpp::Named("icept") = icept_mcmc,
    Rcpp::Named("theta_mc") = theta_mcmc,
    Rcpp::Named("g_beta") = g_beta_store,
    Rcpp::Named("g_gamma") = g_gamma_store,
    Rcpp::Named("g_intercept") = g_intercept
    //Rcpp::Named("sigmasq_sampled") = bmms_t.modules[bmms_t.n_stages-1].sigmasq_sampled
  );
}


//[[Rcpp::export]]
arma::vec reshape_mat(arma::mat X){
  X.reshape(X.n_rows*X.n_cols, 1);
  return X.col(0);
}

//'@export
//[[Rcpp::export]]
Rcpp::List hp_binary_cpp(arma::vec y, arma::cube X, arma::mat X_g, 
                         arma::field<arma::mat> centers,
                         arma::mat mask_forbid,
                         arma::mat Xlocations, // locations of the columns of X_g in the slices of X. marked by ones like mask_forbid
                         double lambda_centers, double lambda_ridge, 
                         int mcmc, int burn, int radius=2,
                         int start_movinglev=0, int partnum=0, 
                         bool save=true, bool save_more_data = true,
                         bool fixsigma = false,
                         double g = -1.0,
                         double g_vs = 1.0,
                         double module_prior_par_vs=1.0){
  
  // X_g is the flattened version of X
  // 
  
  arma::vec Xlocations_vec = reshape_mat(Xlocations);
  arma::uvec Xlocs_ix = arma::find(Xlocations_vec == 1);
  
  int n = y.n_elem;
  double to_from_ratio = 0.0;
  double mhr = 0.0;
  int max_stages = centers.n_elem;
  
  int move_proposed = 0;
  int move_accepted = 0;
  
  int add_proposed = 0;
  int add_accepted = 0;
  int drop_proposed = 0;
  int drop_accepted = 0;
  
  int splitpar_proposed = 0;
  int splitpar_accepted = 0;
  
  arma::vec lambda_mcmc;
  arma::cube splitmask_mcmc;
  arma::field<arma::field<arma::mat>> splitsub_mcmc;
  
  arma::field<arma::cube> theta_mcmc(mcmc-burn);
  arma::field<arma::vec> icept_mcmc(mcmc-burn);
  arma::vec dim_mcmc(mcmc-burn);
  
  if(save_more_data == true){
    splitsub_mcmc = arma::field<arma::field<arma::mat>>(mcmc-burn);
    lambda_mcmc = arma::vec(mcmc-burn);
    splitmask_mcmc = arma::zeros(X.n_rows, X.n_cols, mcmc-burn);
  };
  
  arma::vec z(n);
  arma::mat zsave(n, mcmc-burn);
  
  // initialize
  arma::mat propose_splitmask;
  arma::field<arma::mat> propose_splitsub;
  
  clog << "initalize " << endl;
  ModularLR2D bmms_t = ModularLR2D(y, X, centers, mask_forbid, max_stages, lambda_ridge, fixsigma, 1.0, g);
  int num_levs = bmms_t.n_stages;
  
  z = bmms_t.modules[0].X_flat * bmms_t.modules[0].flatmodel.b;
  
  // vs last module
  arma::vec gamma_start = arma::zeros(X_g.n_cols)+.1;
  arma::vec g_intercept = arma::zeros(mcmc-burn);
  arma::mat g_beta_store = arma::zeros(X_g.n_cols, mcmc-burn);
  arma::mat g_gamma_store = arma::zeros(X_g.n_cols, mcmc-burn);
  double gg_g = g_vs;
  double module_prior_par = module_prior_par_vs;
  
  clog << "prepare vs " << endl;
  arma::vec newprior_filter = arma::zeros(X_g.n_cols);
  PriorVS vsmodule(z, X_g, 
                   newprior_filter,
                   gamma_start, gg_g, module_prior_par, true, 1); // true = binary
  clog << "prepare mcmc " << endl;
  for(unsigned int m = 0; m<mcmc; m++){
    Rcpp::checkUserInterrupt();
    if(m==0){ 
      clog << "starting mcmc" << endl;
    }
    int choices = 3;
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    int rnd_moving_lev = arma::randi<int>(arma::distr_param(start_movinglev, num_levs-1));
    int accepted_proposal = 0;
    try {
      if(move_type == 0){
        //clog << "M";
        move_proposed++;
        
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
        
        propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
        
        ModularLR2D proposed_bayeslm = bmms_t;
        
        //clog << "changing " << endl;
        proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
        
        arma::vec newprior_all = reshape_mat(bmfuncs::cube_sum(bmms_t.theta_sampled, 2));
        arma::vec newprior_filter = newprior_all.elem(Xlocs_ix);
        PriorVS propose_vsmodule(z, X_g, 
                                 newprior_filter, 
                                 gamma_start, gg_g, module_prior_par, true, 0); // true = binary
        
        mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks) + 
          propose_vsmodule.logpost - vsmodule.logpost);
        
        mhr = mhr > 1 ? 1 : mhr;
        
        //clog << "moving mhr " << mhr << endl << "to_from_ratio " << to_from_ratio << endl;
        
        accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
        if(accepted_proposal == 1){
          //clog << "[MOVE SPLIT " << stage << "] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
          proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
          bmms_t = proposed_bayeslm;
          move_accepted++;
        } 
      }
      if(move_type == 1){
        //clog << "A";
        add_proposed ++;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits<100){
          propose_splitmask = split_add2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                          bmms_t.mask_nosplits,
                                          rnd_moving_lev, 
                                          to_from_ratio);
          //cout << propose_splitmask << endl;
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
          //cout << propose_splitsub.row(rnd_moving) << endl;
          ModularLR2D proposed_bayeslm = bmms_t;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          //mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks)) * to_from_ratio * totsplit_prior_mhr;
          
          arma::vec newprior_all = reshape_mat(bmfuncs::cube_sum(bmms_t.theta_sampled, 2));
          arma::vec newprior_filter = newprior_all.elem(Xlocs_ix);
          PriorVS propose_vsmodule(z, X_g, 
                                   newprior_filter, 
                                   gamma_start, gg_g, module_prior_par, true, 0); // true = binary
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks) + 
            propose_vsmodule.logpost - vsmodule.logpost);
          
          mhr = mhr > 1 ? 1 : mhr;
          //cout << mhr << endl;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            //clog << "ADDED " << endl;
            add_accepted ++;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
          } 
          //clog << "add " << totsplit_prior_mhr << endl;
        }
      }
      if(move_type == 2){
        //clog << "D";
        drop_proposed ++;
        int num_splits = bmms_t.modules[rnd_moving_lev].splitmat(rnd_moving_lev).n_rows;
        //clog << num_splits << endl;
        if(num_splits>2){
          
          
          propose_splitmask = split_drop2d(bmms_t.modules[bmms_t.n_stages-1].splitmask, 
                                           bmms_t.mask_nosplits,
                                           rnd_moving_lev,  
                                           to_from_ratio);
          
          
          propose_splitsub = bm2d::splitmask_to_splitsub(propose_splitmask);
          
          ModularLR2D proposed_bayeslm = bmms_t;
          //clog << "changing " << endl;
          proposed_bayeslm.propose_change_module(rnd_moving_lev, propose_splitsub);
          //clog << "done" << endl;
          
          double totsplit_prior_mhr = totsplit_prior2_ratio(proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows,//propose_splitsub(rnd_moving_lev).n_rows, 
                                                            bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows, n, 
                                                            propose_splitsub.n_elem - rnd_moving_lev - 1, //rnd_moving_lev,  
                                                            lambda_centers);
          
          
          arma::vec newprior_all = reshape_mat(bmfuncs::cube_sum(bmms_t.theta_sampled, 2));
          arma::vec newprior_filter = newprior_all.elem(Xlocs_ix);
          PriorVS propose_vsmodule(z, X_g, 
                                   newprior_filter, 
                                   gamma_start, gg_g, module_prior_par, true, 0); // true = binary
          
          mhr = exp(arma::accu(proposed_bayeslm.logliks - bmms_t.logliks) + 
            propose_vsmodule.logpost - vsmodule.logpost);
          
          //clog << "drop: " << to_from_ratio << " " << totsplit_prior_mhr << endl;
          mhr = mhr > 1 ? 1 : mhr;
          //clog << "and this " << endl;
          accepted_proposal = bmrandom::rndpp_discrete({1-mhr, mhr});
          if(accepted_proposal == 1){
            drop_accepted ++;
            //clog << "DROPPED " << proposed_bayeslm.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << " from " << bmms_t.modules[rnd_moving_lev].splitmat[rnd_moving_lev].n_rows << endl;
            proposed_bayeslm.confirm_change_module(rnd_moving_lev, propose_splitsub);
            bmms_t = proposed_bayeslm;
          }
          //clog << "drop " << totsplit_prior_mhr << endl;
          
        } else {
          //clog << "didnt" << endl;
        }
        
      }
      
      num_levs = bmms_t.n_stages;
    } catch(...) {
      clog << "Skipping failed move. Type: (" << move_type << ")" << endl;
    }
    
    //////////////// variable selection on finest scale (Gordon 333)
    arma::vec newprior_all = reshape_mat(bmfuncs::cube_sum(bmms_t.theta_sampled, 2));
    newprior_all = arma::zeros(newprior_all.n_elem);
    arma::vec newprior_filter = newprior_all.elem(Xlocs_ix);
    vsmodule = PriorVS(z, X_g, 
                       newprior_filter,
                       gamma_start, gg_g, module_prior_par, true, 1); // true = binary
    
    arma::mat g_xb = X_g * vsmodule.beta_stored.col(0) + vsmodule.icept_stored(0);
    //clog << "  gamma start" << endl;
    gamma_start = vsmodule.gamma_stored.col(0);
    //clog << gamma_start(1) << endl;
    
    arma::vec w = bmrandom::rpg(arma::ones(y.n_elem), g_xb);
    z = 1.0/w % (y-.5);
    // restart to rebuild prior
    //bmms_t = ModularLR2D(z, X, bmms_t.splitsub, mask_forbid, max_stages, lambda_ridge, fixsigma, 1.0, g);
    
    if(m>burn-1){
      int i = m-burn;
      theta_mcmc(i) = bmms_t.theta_sampled;
      icept_mcmc(i) = bmms_t.icept_sampled;
      dim_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].effective_dimension;
      g_beta_store.col(i) = vsmodule.beta_stored.col(0);
      g_gamma_store.col(i) = vsmodule.gamma_stored.col(0);
      g_intercept(i) = vsmodule.icept_stored(0);// vsmodule.intercept;
      if(save_more_data == true){
        splitsub_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].splitmat;
        //lambda_mcmc(i) = bmms_t.lambda;
        //splitsparam_mcmc(i) = lambda_centers;
        //zsave.col(i) = z;
        splitmask_mcmc.slice(i) = bmms_t.modules[bmms_t.n_stages-1].splitmask;
      };
      //Xflat_mcmc(i) = bmms_t.modules[bmms_t.n_stages-1].X_flat;
    }
    
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        Rcpp::checkUserInterrupt();
        clog << endl << 
          partnum << " " << floor(100.0*(m+0.0)/mcmc) << 
            " " << bmms_t.n_stages << " " << bmms_t.modules[bmms_t.n_stages-1].effective_dimension << 
              " a:" << add_accepted / (add_proposed+0.0) << " d:" << drop_accepted / (drop_proposed+0.0) << " m:" <<
                move_accepted / (move_proposed+0.0) << " #(" << arma::sum(vsmodule.gamma_stored.col(0)) << 
                  ") z " << z.max() << endl;
        if(save){
          bmms_t.modules[bmms_t.n_stages-1].splitmat.save("bmms_centers.temp");
        }
      } 
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("dimension_overall") = dim_mcmc,
    Rcpp::Named("icept") = icept_mcmc,
    Rcpp::Named("theta_mc") = theta_mcmc,
    Rcpp::Named("g_beta") = g_beta_store,
    Rcpp::Named("g_gamma") = g_gamma_store,
    Rcpp::Named("g_intercept") = g_intercept
  );
}





