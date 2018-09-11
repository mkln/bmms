//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#include <string>
#include "image_mcmc_helper.h"
#include "metrop_helper.h"

using namespace std;


// from matrix Sx2 where each row is a 2D split
// and p1, p2 is the grid dimension
// return a masking matrix size p1xp2 with 1 at split locations
//[[Rcpp::export]]
arma::mat splitsub_to_splitmask(const arma::field<arma::mat>& splits, int p1, int p2){
  // given dimensions p1xp2 and the splits
  // returns matrix of zeros + l in split locations
  int lev = splits.n_elem;
  arma::mat mask = arma::zeros(p1, p2);
  for(unsigned int l=0; l<lev; l++){
    for(unsigned int i=0; i<splits(l).n_rows; i++){
      mask(splits(l)(i,0), splits(l)(i,1)) = l+1;
    }
  }
  // lower right corner cannot be used if other than voronoi
  //mask(p1-1, p2-1) = -1;
  return mask;
}

// obtain Sx2 split matrix from a split mask
//[[Rcpp::export]]
arma::field<arma::mat> splitmask_to_splitsub(const arma::mat& splitmask){
  int lev = splitmask.max();
  arma::field<arma::mat> subs(lev);
  for(unsigned int l=0; l<lev; l++){
    arma::uvec split_locs_in_mask = arma::find(splitmask==l+1);
    subs(l) = arma::trans(arma::conv_to<arma::mat>::from( arma::ind2sub(arma::size(splitmask), split_locs_in_mask)));
  }
  return subs;
}



/*
*              FUNCTIONS FOR GROUPING MASKS
*                   AND COARSENING
*/

// from a starting grouping mask, split the relevant region 
// using the new split onesplit. for BLOCKS, not voronoi
arma::mat mask_onesplit(arma::mat startmat, arma::vec onesplit, int seq){
  int p1 = startmat.n_rows;
  int p2 = startmat.n_cols;
  //int maxval = startmat.max();
  arma::mat mask = startmat;
  for(unsigned int i=0; i<p1; i++){
    for(unsigned int j=0; j<p2; j++){
      //if(startmat(i,j) == startmat(onesplit(0), onesplit(1))){
      if(i <= onesplit(0)){
        if(j <= onesplit(1)){
          mask(i,j)=startmat(i,j)+1*pow(10, seq);
        } else {
          mask(i,j)=startmat(i,j)+2*pow(10, seq);
        }
      } else {
        if(j <= onesplit(1)){
          mask(i,j)=startmat(i,j)+3*pow(10, seq);
        } else {
          mask(i,j)=startmat(i,j)+4*pow(10, seq);
        }
      }
      //}
    } 
  }
  return(mask-mask.min());
}

// make a grouping mask from split matrix
// a grouping max assigns each element of the grid to a numbered group
arma::mat splitsub_to_groupmask_blocks(const arma::mat& splits, int p1, int p2){
  // splits is a nsplit x 2 matrix
  arma::mat splitted = arma::zeros(p1, p2);
  for(unsigned int i=0; i<splits.n_rows; i++){
    if((splits(i,0)==p1-1) & (splits(i,1)==p2-1)){
      cout << splits.t() << endl;
      throw std::invalid_argument("edge splits not allowed: nothing to split");
    } else {
      if((splits(i,0)>p1-1) || (splits(i,1)>p2-1) || (splits(i,0)<0) || (splits(i,1)<0)){
        throw std::invalid_argument("one split outside splitting region");
      } else {
        splitted = mask_onesplit(splitted, splits.row(i).t(), i);
      }
    }
  }
  return(splitted);
}

//[[Rcpp::export]]
arma::mat row_intersection(const arma::mat& mat1, const arma::mat& mat2){
  arma::mat inter = -1*arma::zeros(mat1.n_rows<mat2.n_rows? mat1.n_rows : mat2.n_rows, 2);
  int c=0;
  for(unsigned int i=0; i<mat1.n_rows; i++){
    for(unsigned int j=0; j<mat2.n_rows; j++){
      if(arma::approx_equal(mat1.row(i), mat2.row(j), "absdiff", 0.002)){
        inter.row(c) = mat1.row(i);
        c++;
      }
    }
  }
  if(c>0){
    return inter.rows(0,c-1);
  } else {
    return arma::zeros(0,2);
  }
}

//[[Rcpp::export]]
arma::mat row_difference(arma::mat mat1, const arma::mat& mat2){
  arma::mat diff = -1*arma::zeros(mat1.n_rows, 2);
  int c=0;
  for(unsigned int i=0; i<mat1.n_rows; i++){
    bool foundit = false;
    for(unsigned int j=0; j<mat2.n_rows; j++){
      if(arma::approx_equal(mat1.row(i), mat2.row(j), "absdiff", 0.002)){
        foundit = true;
      }
    }
    if(foundit == false){
      diff.row(c) = mat1.row(i);
      c++;
    }
  }
  if(c>0){
    return diff.rows(0,c-1);
  } else {
    return arma::zeros(0,2);
  }
}

//[[Rcpp::export]]
arma::field<arma::mat> splits_augmentation(arma::field<arma::mat> splits){
  arma::field<arma::mat> splits_augment(splits.n_elem);
  // append previous splits to currents
  splits_augment(0) = splits(0);
  for(unsigned int i=1; i<splits.n_elem; i++){
    splits_augment(i) = arma::join_vert(splits_augment(i-1), splits(i));
  }
  return splits_augment;
}


//with voronoi tessellation
//[[Rcpp::export]]
arma::mat splitsub_to_groupmask(arma::field<arma::mat> splits, int p1, int p2){
  // splits is a nsplit x 2 matrix
  arma::vec distances = arma::ones(splits(0).n_rows);
  arma::mat splitted = arma::zeros(p1, p2);
  // level 0
  for(unsigned int i=0; i<p1; i++){
    for(unsigned int j=0; j<p2; j++){
      for(unsigned int s=0; s<splits(0).n_rows; s++){
        distances(s) = pow(0.0+i-splits(0)(s,0), 2) + pow(0.0+j-splits(0)(s,1), 2);
      }
      //clog << distances << endl;
      splitted(i, j) = distances.index_min();
    } 
  }
  
  splits = splits_augmentation(splits);
  
  //cout << splitted << endl;
  // other levels
  int lev = splits.n_elem;
  for(unsigned int l=1; l<splits.n_elem; l++){
    //clog << "more than one level! " << endl;
    // subset the search on the points in the groupmask that
    // belong to the same group in the previous level
    int splits_at_prev_lev = splits(l-1).n_rows;
    //clog << "splits at previous level " << splits_at_prev_lev << endl;
    // loop over possible values of previous levels
    for(unsigned int s=0; s<splits_at_prev_lev; s++){
      //clog << "prev split " << splits(l-1).row(s) << endl;
      // for each split at this level,
      // subset matrix elements with same value of split
      arma::uvec locs = arma::find(splitted == splitted( splits(l-1)(s,0), splits(l-1)(s,1) ));
      // relevant splits are only those that are also in the same area
      //arma::mat rett = arma::intersect(arma::ind2sub(arma::size(splitted), locs), arma::conv_to<arma::umat>::from(splits(l)));
      arma::mat all_locs_subs = arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(splitted), locs));
      //clog << "locs with same value as " << splits(l-1).row(s) << endl;
      //clog << all_locs_subs.t() << endl;
      arma::mat relevant = row_intersection(all_locs_subs.t(), splits(l));
      //clog << "relevant splits: " << relevant << endl;
      // distance of subset points from all relevant splits at this level
      distances = arma::ones(relevant.n_rows);
      if(distances.n_elem > 0){
        for(unsigned int i=0; i<locs.n_elem; i++){ // like for i, for j, but vectorized
          arma::uvec inde = arma::ind2sub(arma::size(splitted), locs(i));
          //clog << "location " << inde << endl;
          for(unsigned int r=0; r<relevant.n_rows; r++){
            distances(r) = pow(0.0+inde(0) - relevant(r,0), 2) + pow(0.0+ inde(1) - relevant(r,1), 2);
          }
          splitted(inde(0), inde(1)) += (1+distances.index_min())*pow(10,l);
          //clog << "element in subset " << inde.t() << endl;
          //clog << "min dist with split index " << distances.index_min() << endl;
        } 
      }
    }
  }
  return(splitted);
}

// extract region numbers (labels) from a grouping mask
//[[Rcpp::export]]
arma::vec mat_unique(const arma::mat& A){
  arma::uvec uvals = arma::find_unique(A);
  return(A.elem(uvals));
}

// returns a matrix where all unselected regions are set to 0
//[[Rcpp::export]]
arma::mat mask_oneval(const arma::mat& A, const arma::mat& mask, int val){
  arma::uvec uvals = arma::find(mask != val);
  arma::mat retmat = A;
  retmat.elem(uvals).fill(0.0);
  return(retmat);
}

// using a grouping mask, sum values in a matrix corresponding to 
// the same group (2d coarsening operation)
//[[Rcpp::export]]
double mask_oneval_sum(const arma::mat& A, const arma::mat& mask, int val){
  arma::uvec uvals = arma::find(mask == val);
  return(arma::accu(A.elem(uvals)));
}

// same as mask_oneval_sum but for a cube slice
// cube here will be the cube-X to be transformed in matrix-X
//[[Rcpp::export]]
double mask_cube_slice(const arma::cube& C, int slice, const arma::mat& mask, int val){
  arma::uvec uvals = arma::find(mask == val);
  return(arma::accu(C.slice(slice).elem(uvals)));
}

// transform a regressor matrix to a vector using grouping mask as coarsening
//[[Rcpp::export]]
arma::vec mat_to_vec_by_region(const arma::mat& A, const arma::mat& mask, arma::vec unique_regions){
  //arma::vec unique_regions = mat_unique(mask);
  int n_unique_regions = unique_regions.n_elem;
  arma::vec vectorized_mat = arma::zeros(n_unique_regions);
  for(unsigned int r=0; r<n_unique_regions; r++){
    vectorized_mat(r) = mask_oneval_sum(A, mask, unique_regions(r));
  }
  return(vectorized_mat);
}

// using a grouping mask, transforma a cube to a matrix
//[[Rcpp::export]]
arma::mat cube_to_mat_by_region(const arma::cube& C, const arma::mat& mask, arma::vec unique_regions){
  // cube is assumed dimension (p1, p2, n)
  int n_unique_regions = unique_regions.n_elem;
  arma::mat matricized_cube = arma::zeros(C.n_slices, n_unique_regions);
  for(unsigned int i=0; i<C.n_slices; i++){
    // every row in the cube is a matrix observation
    for(unsigned int r=0; r<n_unique_regions; r++){
      // we transform every matrix into a vector by region
      matricized_cube(i,r) = mask_cube_slice(C, i, mask, unique_regions(r));
    }
  }
  return(matricized_cube);
}

// given a vectorized beta vector, a vector of labels, and a grouping mask
// return a matrix of size size(mask) filling it with beta using regions
//[[Rcpp::export]]
arma::mat unmask_vector(const arma::vec& beta, const arma::vec& regions, const arma::mat& mask){
  // takes a vector and fills a matrix of dim size(mask) using regions
  arma::mat unmasked_vec = arma::zeros(arma::size(mask));
  for(unsigned int r=0; r<regions.n_elem; r++){
    unmasked_vec.elem(arma::find(mask == regions(r))).fill(beta(r));
  }
  return(unmasked_vec);
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
  splitmask = splitsub_to_splitmask(splitmat, p1, p2);
  mask_nosplits = mask_forbid;
  
  groupmask = splitsub_to_groupmask(splits, p1, p2);
  regions = mat_unique(groupmask);
  Xcube = X2d;
  X_flat = cube_to_mat_by_region(X2d, groupmask, regions);
  //clog << col_sums(X_flat) << endl;
  effective_dimension = X_flat.n_cols;
  flatmodel = BayesLM(y, X_flat, lambda, binary); 
  //flatmodel = VSModule(y, X_flat, lambda, binary);
    
  
  //mu_post = unmask_vector(flatmodel.mu, regions, groupmask);
  //Sigma_post = flatmodel.Sigma;
  sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
  a_post = flatmodel.alpha_n;
  b_post = flatmodel.beta_n;
  
  icept_sampled = flatmodel.icept;
  beta_sampled = unmask_vector(flatmodel.b, regions, groupmask);
  sigmasq_sampled = flatmodel.sigmasq;
  
  Xb = flatmodel.reg_mean;
  residuals = y - Xb;
  //clog << "created BLM2D" << endl;
}

void BayesLM2D::change_splits(arma::field<arma::mat>& splits){
  splitmat = splits;
  splitmask = splitsub_to_splitmask(splitmat, p1, p2);
  
  groupmask = splitsub_to_groupmask(splits, p1, p2);
  regions = mat_unique(groupmask);
  
  X_flat = cube_to_mat_by_region(Xcube, groupmask, regions);
  effective_dimension = X_flat.n_cols;
  
  flatmodel = BayesLM(y, X_flat, lambda, binary); 
  //flatmodel = VSModule(y, X_flat, lambda, binary);
  
  //mu_post = unmask_vector(flatmodel.mu, regions, groupmask);
  //Sigma_post = flatmodel.Sigma;
  sigmasq_post_mean = flatmodel.beta_n/(flatmodel.alpha_n+1);
  a_post = flatmodel.alpha_n;
  b_post = flatmodel.beta_n;
  
  icept_sampled = flatmodel.icept;
  beta_sampled = unmask_vector(flatmodel.b, regions, groupmask);
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

//[[Rcpp::export]]
arma::field<arma::mat> load_splits(int maxlevs, std::string sname){
  arma::field<arma::mat> splits(maxlevs);
  splits.load(sname);
  splits.print();
  return splits;
}


//[[Rcpp::export]]
arma::field<arma::mat> merge_splits(arma::field<arma::mat>& old_splits, arma::field<arma::mat> new_splits){
  arma::field<arma::mat> splits(old_splits.n_elem);
  for(unsigned int s = 0; s<new_splits.n_elem; s++){
    splits(s) = arma::join_vert(old_splits(s), new_splits(s));
  }
  return splits;
}

//[[Rcpp::export]]
double gammaprior_mhr(double new_val, double old_val, double alpha, double beta){
  return (alpha-1) * (log(new_val) - log(old_val)) - 1.0/beta * (new_val - old_val);
}



//[[Rcpp::export]]
arma::mat cube_mean(arma::cube X, int dim){
  return arma::mean(X, dim);
}
//[[Rcpp::export]]
arma::mat cube_sum(arma::cube X, int dim){
  return arma::sum(X, dim);
}

