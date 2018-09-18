//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::interfaces(cpp)]]

#include "bmms_common.h"

using namespace std;

std::random_device rd;
std::mt19937 mt(rd());
int basis_type = 1;

arma::mat X2Dgrid(arma::vec x1, arma::vec x2){
  arma::mat rr = arma::zeros(x1.n_elem*x2.n_elem, 2);
  for(unsigned int i=0; i<x1.n_elem; i++){
    for(unsigned int j=0; j<x2.n_elem; j++){
      rr(i*x2.n_elem+j, 0) = x1(i);
      rr(i*x2.n_elem+j, 1) = x2(j);
    }
  }
  return(rr);
}


double rndpp_bern(double p){
  double run = arma::randu();
  if(run < p){ 
    return 1.0;
  } else {
    return 0.0;
  }
}


double split_struct_ratio(arma::vec prop_split, arma::vec orig_split, int p, double param){
  //return exp(-param*(gini(prop_split,p) + gini(orig_split,p)));
  return 1.0; 
}


arma::vec bmms_setdiff(arma::vec& x, arma::vec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  
  return arma::conv_to<arma::vec>::from(out);
}


int rndpp_unif_int(int max){
  std::uniform_int_distribution<int> d(0,max);
  return d(mt);
}

int rndpp_discrete(arma::vec probs){
  std::discrete_distribution<> d(probs.begin(), probs.end());
  return d(mt);
}

double rndpp_gamma(double alpha, double beta) 
{
  // E(X) = alpha * beta
  std::gamma_distribution<double> dist(alpha, beta);
  return dist(mt);
}

double rndpp_normal(double mean, double sigma) 
{
  std::normal_distribution<double> dist(mean, sigma);
  return dist(mt);
}

arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  int dimension = arma::size(mean)(0);
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  for(int i=0; i<n; i++){
    //for(int j=0; j<dimension; j++){
    //  xtemp(j) = rndpp_normal(0.0, 1.0, mt);
    //}
    xtemp = Rcpp::rnorm(dimension, 0.0, 1.0);
    //clog << arma::det(sigma) << endl;
    outmat.row(i) = (mean + cholsigma * xtemp).t();
  }
  return outmat;
}


arma::vec nonzeromean(arma::mat mat_mcmc){
  arma::vec result = arma::zeros(mat_mcmc.n_rows);
  for(unsigned int j=0; j<mat_mcmc.n_rows; j++){
    arma::vec thisrow = mat_mcmc.row(j).t();
    arma::vec nnzero = thisrow(arma::find(thisrow));
    result(j) = nnzero.n_elem > 0 ? arma::mean(nnzero) : 0.0;
  }
  return result;
}

arma::vec col_eq_check(arma::mat A){
  arma::vec is_same_as = arma::ones(A.n_cols) * -1;
  
  for(unsigned int i1=0; (i1<A.n_cols); i1++){
    for(unsigned int i2=A.n_cols-1; i2 > i1; i2--){
      if(approx_equal(A.col(i1), A.col(i2), "absdiff", 1e-10)){
        if(is_same_as(i2) == -1) { 
          is_same_as(i2) = i1;
        }
      } 
    }
  }
  return is_same_as;
}

arma::vec col_sums(const arma::mat& matty){
  return arma::sum(matty, 0).t();
}


arma::mat single_split(arma::mat Jcoarse, int where, int p){
  int which_row = where;
  int which_col = arma::conv_to<int>::from(arma::find(Jcoarse.row(which_row), 1, "first"));
  
  arma::mat slice_upleft = Jcoarse.submat(0, which_col, which_row, which_col);
  arma::mat slice_downright = Jcoarse.submat(which_row+1, which_col, p-1, which_col);
  arma::mat upright = arma::zeros(which_row+1, 1);
  arma::mat downleft = arma::zeros(p-which_row-1, 1);
  arma::mat up = arma::join_rows(slice_upleft, upright);
  arma::mat down = arma::join_rows(downleft, slice_downright);
  arma::mat splitted = arma::join_cols(up, down); 
  return splitted;
}

arma::mat multi_split(arma::mat Jcoarse, arma::vec where, int p){
  //int p = arma::accu(Jcoarse);
  //int c = Jcoarse.n_cols;
  arma::vec excluding(1);
  arma::vec orig_cols_splitted = arma::zeros(Jcoarse.n_cols);
  arma::mat slice_upleft, slice_downright, upright, downleft, up, down, splitted;
  
  arma::mat splitting_mat = Jcoarse;
  arma::mat temp(Jcoarse.n_rows, 0);
  //cout << Jcoarse << endl;
  for(unsigned int w=0; w<where.n_elem; w++){
    unsigned int which_row = where(w);
    //cout << "multi_split: inside loop. splitting row " << which_row << " of " << endl;
    //cout << Jcoarse << endl; 
    int which_col = arma::conv_to<int>::from(arma::find(Jcoarse.row(which_row), 1, "first"));
    
    //cout << "splitting at row " << which_row << endl;
    //cout << "corresponding to " << which_col << " in the original matrix" << endl;
    //cout << "current status of splitted cols " << orig_cols_splitted.t() << endl;
    
    if(orig_cols_splitted(which_col) == 1){
      // we had already split this column before
      // this means we are splitting a column from the matrix of splits
      //cout << "we have already split this column before, hence taking this matrix " << endl << temp << endl;
      which_col = arma::conv_to<unsigned int>::from(arma::find(temp.row(which_row), 1, "first"));
      //cout << "hence the column we need to look at is actually " << which_col << endl;
      if((which_row>=temp.n_rows-1) || (temp(which_row+1, which_col) != 0)){
        splitted = single_split(temp, which_row, p);
        //cout << "and we obtain the following matrix " << endl << splitted << endl;
        excluding(0) = which_col;
        temp = exclude(temp, excluding);
        temp = arma::join_horiz(temp, splitted);
        //cout << "status up to now: " << endl << temp << endl;
      } //else {
      // cout << "ineffective split " << endl;
      //}
    } else {
      // first time we split this column on the original matrix
      //cout << "did not split this column before" << endl;
      orig_cols_splitted(which_col) = 1;
      //cout << which_row << " : " << which_col << endl; //" [ " << priorprob.t() <<  " ] " << endl;
      if((which_row>=Jcoarse.n_rows-1) || (Jcoarse(which_row+1, which_col) != 0)){
        //cout << "starting from the original " << endl << Jcoarse << endl;
        splitted = single_split(Jcoarse, which_row, p);
        //cout << "we obtain the following matrix " << endl << splitted << endl;
        temp = arma::join_horiz(temp, splitted);
        //cout << "status up to now: " << endl << temp << endl;
      } //else {
      //cout << "ineffective split " << endl;
      //}
    }
    //cout << "===================================" << endl;
  }
  //cout << test1 << endl << test2 << endl << test << endl;
  return temp;
}


arma::vec split_fix(arma::field<arma::vec>& in_splits, int stage){
  arma::vec splits_if_any = in_splits(stage)(arma::find(in_splits(stage)!=-1));
  return splits_if_any;
  //cout << "OUT SPLITS " << in_splits << endl;
}

arma::field<arma::vec> stage_fix(arma::field<arma::vec>& in_splits){
  //cout << "IN SPLITS " << in_splits << endl;
  //cout << "theoretical number of stages : " << in_splits.n_elem << endl;
  int n_stages = 0;
  for(unsigned int i=0; i<in_splits.n_elem; i++){
    if(in_splits(i).n_elem > 0){
      bool actual_stage = false;
      for(unsigned int j=0; j<in_splits(i).n_elem; j++){
        if(in_splits(i)(j)>-1){
          actual_stage = true;
        }
      }
      n_stages = actual_stage ? (n_stages+1) : n_stages;
    }
  }
  //cout << "actual number of stages ; " << n_stages << endl;
  int loc = 0;
  arma::field<arma::vec> split_seq(n_stages);
  for(unsigned int i=0; i<in_splits.n_elem; i++){
    if(in_splits(i).n_elem > 0){
      arma::vec splits_if_any = in_splits(i)(arma::find(in_splits(i)!=-1));
      if(splits_if_any.n_elem > 0){
        split_seq(loc) = splits_if_any;
        loc++;
      }
    }
  }
  return split_seq;
  //cout << "OUT SPLITS " << in_splits << endl;
}


arma::mat exclude(arma::mat test, arma::vec excl){
  arma::vec keepers = arma::ones(test.n_cols);
  for(unsigned int e=0; e<excl.n_elem; e++){
    keepers(excl(e)) = 0;
  }
  //cout << "exclude cols " << endl;
  return test.cols(arma::find(keepers));
}
