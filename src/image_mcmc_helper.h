//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef image_voronoi
#define image_voronoi

#include "linear_conjugate.h"
#include "metrop_helper.h"
//#include "bmms_0906_varsel.h"
#include <RcppArmadillo.h>

// from matrix Sx2 where each row is a 2D split
// and p1, p2 is the grid dimension
// return a masking matrix size p1xp2 with 1 at split locations
arma::mat splitsub_to_splitmask(const arma::field<arma::mat>& splits, int p1, int p2);
// obtain Sx2 split matrix from a split mask
arma::field<arma::mat> splitmask_to_splitsub(const arma::mat& splitmask);
/*
*              FUNCTIONS FOR GROUPING MASKS
*                   AND COARSENING
*/

// from a starting grouping mask, split the relevant region 
// using the new split onesplit. for BLOCKS, not voronoi
arma::mat mask_onesplit(arma::mat startmat, arma::vec onesplit, int seq);
// make a grouping mask from split matrix
// a grouping max assigns each element of the grid to a numbered group
arma::mat splitsub_to_groupmask_blocks(const arma::mat& splits, int p1, int p2);

arma::mat row_intersection(const arma::mat& mat1, const arma::mat& mat2);

arma::mat row_difference(arma::mat mat1, const arma::mat& mat2);

arma::field<arma::mat> splits_augmentation(arma::field<arma::mat> splits);

//with voronoi tessellation

arma::mat splitsub_to_groupmask(arma::field<arma::mat> splits, int p1, int p2);
// extract region numbers (labels) from a grouping mask

arma::vec mat_unique(const arma::mat& A);
// returns a matrix where all unselected regions are set to 0

arma::mat mask_oneval(const arma::mat& A, const arma::mat& mask, int val);
// using a grouping mask, sum values in a matrix corresponding to 
// the same group (2d coarsening operation)

double mask_oneval_sum(const arma::mat& A, const arma::mat& mask, int val);
// same as mask_oneval_sum but for a cube slice
// cube here will be the cube-X to be transformed in matrix-X

double mask_cube_slice(const arma::cube& C, int slice, const arma::mat& mask, int val);
// transform a regressor matrix to a vector using grouping mask as coarsening

arma::vec mat_to_vec_by_region(const arma::mat& A, const arma::mat& mask, arma::vec unique_regions);
// using a grouping mask, transforma a cube to a matrix

arma::mat cube_to_mat_by_region(const arma::cube& C, const arma::mat& mask, arma::vec unique_regions);

// given a vectorized beta vector, a vector of labels, and a grouping mask
// return a matrix of size size(mask) filling it with beta using regions

arma::mat unmask_vector(const arma::vec& beta, const arma::vec& regions, const arma::mat& mask);


double blm_marglik(arma::vec& y, arma::mat& mean_post, arma::mat& inv_var_post, double a, double b);

class BayesLM2D{
public:
  
  bool binary;
  
  arma::mat X_flat;
  arma::cube Xcube;
  
  arma::vec y;
  
  int p1, p2;
  
  arma::field<arma::mat> splitmat;
  arma::mat splitmask;
  
  arma::mat Beta;
  arma::mat groupmask;
  arma::vec regions;
  
  double lambda;
  
  BayesLM flatmodel; //***
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
  
  BayesLM2D(arma::vec&, arma::cube&, arma::field<arma::mat>&, arma::mat&, double, bool);
};

class ModularLR2D {
public:
  
  bool binary;
  
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
  arma::cube theta_sampled;
  arma::mat mask_nosplits;
  
  arma::vec Xb_sum;
  //void redo();
  
  ModularLR2D(const arma::vec&, const arma::cube&, const arma::field<arma::mat>&, arma::mat&, int, double);
  ModularLR2D(const arma::vec&, const arma::cube&, const arma::field<arma::mat>&, arma::mat&, int, double, bool);
};

arma::field<arma::mat> load_splits(int maxlevs);

arma::field<arma::mat> merge_splits(arma::field<arma::mat>& old_splits, arma::field<arma::mat> new_splits);
double gammaprior_mhr(double new_val, double old_val, double alpha=100, double beta=.5);

arma::mat cube_mean(arma::cube X, int dim);
arma::mat cube_sum(arma::cube X, int dim);

#endif