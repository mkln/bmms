// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// soi_cpp
Rcpp::List soi_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> splits, arma::mat mask_forbid, double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius, int start_movinglev, int partnum, bool save, bool save_splitmask);
RcppExport SEXP _bmms_soi_cpp(SEXP ySEXP, SEXP XSEXP, SEXP splitsSEXP, SEXP mask_forbidSEXP, SEXP lambda_centersSEXP, SEXP lambda_ridgeSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP radiusSEXP, SEXP start_movinglevSEXP, SEXP partnumSEXP, SEXP saveSEXP, SEXP save_splitmaskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mask_forbid(mask_forbidSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_centers(lambda_centersSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_ridge(lambda_ridgeSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< int >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< int >::type start_movinglev(start_movinglevSEXP);
    Rcpp::traits::input_parameter< int >::type partnum(partnumSEXP);
    Rcpp::traits::input_parameter< bool >::type save(saveSEXP);
    Rcpp::traits::input_parameter< bool >::type save_splitmask(save_splitmaskSEXP);
    rcpp_result_gen = Rcpp::wrap(soi_cpp(y, X, splits, mask_forbid, lambda_centers, lambda_ridge, mcmc, burn, radius, start_movinglev, partnum, save, save_splitmask));
    return rcpp_result_gen;
END_RCPP
}
// soi_binary_cpp
Rcpp::List soi_binary_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> centers, arma::mat mask_forbid, double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius, int start_movinglev, int partnum, bool save, bool save_splitmask, bool fixsigma);
RcppExport SEXP _bmms_soi_binary_cpp(SEXP ySEXP, SEXP XSEXP, SEXP centersSEXP, SEXP mask_forbidSEXP, SEXP lambda_centersSEXP, SEXP lambda_ridgeSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP radiusSEXP, SEXP start_movinglevSEXP, SEXP partnumSEXP, SEXP saveSEXP, SEXP save_splitmaskSEXP, SEXP fixsigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type centers(centersSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mask_forbid(mask_forbidSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_centers(lambda_centersSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_ridge(lambda_ridgeSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< int >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< int >::type start_movinglev(start_movinglevSEXP);
    Rcpp::traits::input_parameter< int >::type partnum(partnumSEXP);
    Rcpp::traits::input_parameter< bool >::type save(saveSEXP);
    Rcpp::traits::input_parameter< bool >::type save_splitmask(save_splitmaskSEXP);
    Rcpp::traits::input_parameter< bool >::type fixsigma(fixsigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(soi_binary_cpp(y, X, centers, mask_forbid, lambda_centers, lambda_ridge, mcmc, burn, radius, start_movinglev, partnum, save, save_splitmask, fixsigma));
    return rcpp_result_gen;
END_RCPP
}
// index_to_subscript
arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m);
RcppExport SEXP _bmms_index_to_subscript(SEXP indexSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type index(indexSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(index_to_subscript(index, m));
    return rcpp_result_gen;
END_RCPP
}
// load_splits
arma::field<arma::mat> load_splits(int maxlevs, std::string sname);
RcppExport SEXP _bmms_load_splits(SEXP maxlevsSEXP, SEXP snameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type maxlevs(maxlevsSEXP);
    Rcpp::traits::input_parameter< std::string >::type sname(snameSEXP);
    rcpp_result_gen = Rcpp::wrap(load_splits(maxlevs, sname));
    return rcpp_result_gen;
END_RCPP
}
// sof
Rcpp::List sof(arma::vec& y, arma::mat& X, int max_stages, unsigned int mcmc, unsigned int burn, double lambda, double ain, double bin, bool silent);
RcppExport SEXP _bmms_sof(SEXP ySEXP, SEXP XSEXP, SEXP max_stagesSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP lambdaSEXP, SEXP ainSEXP, SEXP binSEXP, SEXP silentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type max_stages(max_stagesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type ain(ainSEXP);
    Rcpp::traits::input_parameter< double >::type bin(binSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    rcpp_result_gen = Rcpp::wrap(sof(y, X, max_stages, mcmc, burn, lambda, ain, bin, silent));
    return rcpp_result_gen;
END_RCPP
}
// sofk
Rcpp::List sofk(const arma::vec& y, const arma::mat& X, const arma::field<arma::vec>& start_splits, unsigned int mcmc, unsigned int burn, double lambda, double ain, double bin, int ii, int ll, bool silent);
RcppExport SEXP _bmms_sofk(SEXP ySEXP, SEXP XSEXP, SEXP start_splitsSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP lambdaSEXP, SEXP ainSEXP, SEXP binSEXP, SEXP iiSEXP, SEXP llSEXP, SEXP silentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type start_splits(start_splitsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type ain(ainSEXP);
    Rcpp::traits::input_parameter< double >::type bin(binSEXP);
    Rcpp::traits::input_parameter< int >::type ii(iiSEXP);
    Rcpp::traits::input_parameter< int >::type ll(llSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    rcpp_result_gen = Rcpp::wrap(sofk(y, X, start_splits, mcmc, burn, lambda, ain, bin, ii, ll, silent));
    return rcpp_result_gen;
END_RCPP
}
// sofk_binary
Rcpp::List sofk_binary(const arma::vec& y, const arma::mat& X, arma::field<arma::vec> start_splits, unsigned int mcmc, unsigned int burn, double lambda, int ii, int ll, bool silent);
RcppExport SEXP _bmms_sofk_binary(SEXP ySEXP, SEXP XSEXP, SEXP start_splitsSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP lambdaSEXP, SEXP iiSEXP, SEXP llSEXP, SEXP silentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type start_splits(start_splitsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type ii(iiSEXP);
    Rcpp::traits::input_parameter< int >::type ll(llSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    rcpp_result_gen = Rcpp::wrap(sofk_binary(y, X, start_splits, mcmc, burn, lambda, ii, ll, silent));
    return rcpp_result_gen;
END_RCPP
}
// bmms_base
Rcpp::List bmms_base(arma::vec& y, arma::mat& X, double g, int mcmc, int burn, arma::field<arma::vec> splits, bool silent);
RcppExport SEXP _bmms_bmms_base(SEXP ySEXP, SEXP XSEXP, SEXP gSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP splitsSEXP, SEXP silentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type g(gSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    rcpp_result_gen = Rcpp::wrap(bmms_base(y, X, g, mcmc, burn, splits, silent));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bmms_soi_cpp", (DL_FUNC) &_bmms_soi_cpp, 13},
    {"_bmms_soi_binary_cpp", (DL_FUNC) &_bmms_soi_binary_cpp, 14},
    {"_bmms_index_to_subscript", (DL_FUNC) &_bmms_index_to_subscript, 2},
    {"_bmms_load_splits", (DL_FUNC) &_bmms_load_splits, 2},
    {"_bmms_sof", (DL_FUNC) &_bmms_sof, 9},
    {"_bmms_sofk", (DL_FUNC) &_bmms_sofk, 11},
    {"_bmms_sofk_binary", (DL_FUNC) &_bmms_sofk_binary, 9},
    {"_bmms_bmms_base", (DL_FUNC) &_bmms_bmms_base, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_bmms(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
