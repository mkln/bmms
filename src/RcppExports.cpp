// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

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
// sofk
Rcpp::List sofk(const arma::vec& yin, const arma::mat& X, const arma::field<arma::vec>& start_splits, unsigned int mcmc, unsigned int burn, double lambda, double ain, double bin, int ii, int ll, bool onesigma, bool silent, double gin, double structpar, bool trysmooth);
RcppExport SEXP _bmms_sofk(SEXP yinSEXP, SEXP XSEXP, SEXP start_splitsSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP lambdaSEXP, SEXP ainSEXP, SEXP binSEXP, SEXP iiSEXP, SEXP llSEXP, SEXP onesigmaSEXP, SEXP silentSEXP, SEXP ginSEXP, SEXP structparSEXP, SEXP trysmoothSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type yin(yinSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type start_splits(start_splitsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type ain(ainSEXP);
    Rcpp::traits::input_parameter< double >::type bin(binSEXP);
    Rcpp::traits::input_parameter< int >::type ii(iiSEXP);
    Rcpp::traits::input_parameter< int >::type ll(llSEXP);
    Rcpp::traits::input_parameter< bool >::type onesigma(onesigmaSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    Rcpp::traits::input_parameter< double >::type gin(ginSEXP);
    Rcpp::traits::input_parameter< double >::type structpar(structparSEXP);
    Rcpp::traits::input_parameter< bool >::type trysmooth(trysmoothSEXP);
    rcpp_result_gen = Rcpp::wrap(sofk(yin, X, start_splits, mcmc, burn, lambda, ain, bin, ii, ll, onesigma, silent, gin, structpar, trysmooth));
    return rcpp_result_gen;
END_RCPP
}
// sofk_binary
Rcpp::List sofk_binary(const arma::vec& yin, const arma::mat& X, const arma::field<arma::vec>& start_splits, unsigned int mcmc, unsigned int burn, double lambda, double ain, double bin, int ii, int ll, bool onesigma, bool silent, double gin, double structpar, bool trysmooth);
RcppExport SEXP _bmms_sofk_binary(SEXP yinSEXP, SEXP XSEXP, SEXP start_splitsSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP lambdaSEXP, SEXP ainSEXP, SEXP binSEXP, SEXP iiSEXP, SEXP llSEXP, SEXP onesigmaSEXP, SEXP silentSEXP, SEXP ginSEXP, SEXP structparSEXP, SEXP trysmoothSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type yin(yinSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type start_splits(start_splitsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type ain(ainSEXP);
    Rcpp::traits::input_parameter< double >::type bin(binSEXP);
    Rcpp::traits::input_parameter< int >::type ii(iiSEXP);
    Rcpp::traits::input_parameter< int >::type ll(llSEXP);
    Rcpp::traits::input_parameter< bool >::type onesigma(onesigmaSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    Rcpp::traits::input_parameter< double >::type gin(ginSEXP);
    Rcpp::traits::input_parameter< double >::type structpar(structparSEXP);
    Rcpp::traits::input_parameter< bool >::type trysmooth(trysmoothSEXP);
    rcpp_result_gen = Rcpp::wrap(sofk_binary(yin, X, start_splits, mcmc, burn, lambda, ain, bin, ii, ll, onesigma, silent, gin, structpar, trysmooth));
    return rcpp_result_gen;
END_RCPP
}
// bmms_base
Rcpp::List bmms_base(arma::vec& y, arma::mat& X, double sigmasq, double g, int mcmc, int burn, arma::field<arma::vec> splits, bool silent);
RcppExport SEXP _bmms_bmms_base(SEXP ySEXP, SEXP XSEXP, SEXP sigmasqSEXP, SEXP gSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP splitsSEXP, SEXP silentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< double >::type g(gSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    rcpp_result_gen = Rcpp::wrap(bmms_base(y, X, sigmasq, g, mcmc, burn, splits, silent));
    return rcpp_result_gen;
END_RCPP
}
// bmms_debug
int bmms_debug(arma::vec& y, arma::mat& X, double sigmasq, double g, int mcmc, int burn, arma::field<arma::vec> splits, bool silent);
RcppExport SEXP _bmms_bmms_debug(SEXP ySEXP, SEXP XSEXP, SEXP sigmasqSEXP, SEXP gSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP splitsSEXP, SEXP silentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< double >::type g(gSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    rcpp_result_gen = Rcpp::wrap(bmms_debug(y, X, sigmasq, g, mcmc, burn, splits, silent));
    return rcpp_result_gen;
END_RCPP
}
// bmms_vs
Rcpp::List bmms_vs(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, const arma::field<arma::vec>& starting, int mcmc_in, arma::vec gg, arma::vec module_prior_par, bool binary);
RcppExport SEXP _bmms_bmms_vs(SEXP y_inSEXP, SEXP Xall_inSEXP, SEXP startingSEXP, SEXP mcmc_inSEXP, SEXP ggSEXP, SEXP module_prior_parSEXP, SEXP binarySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y_in(y_inSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type Xall_in(Xall_inSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type starting(startingSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc_in(mcmc_inSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gg(ggSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type module_prior_par(module_prior_parSEXP);
    Rcpp::traits::input_parameter< bool >::type binary(binarySEXP);
    rcpp_result_gen = Rcpp::wrap(bmms_vs(y_in, Xall_in, starting, mcmc_in, gg, module_prior_par, binary));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bmms_load_splits", (DL_FUNC) &_bmms_load_splits, 2},
    {"_bmms_soi_cpp", (DL_FUNC) &_bmms_soi_cpp, 13},
    {"_bmms_soi_binary_cpp", (DL_FUNC) &_bmms_soi_binary_cpp, 14},
    {"_bmms_sofk", (DL_FUNC) &_bmms_sofk, 15},
    {"_bmms_sofk_binary", (DL_FUNC) &_bmms_sofk_binary, 15},
    {"_bmms_bmms_base", (DL_FUNC) &_bmms_bmms_base, 8},
    {"_bmms_bmms_debug", (DL_FUNC) &_bmms_bmms_debug, 8},
    {"_bmms_bmms_vs", (DL_FUNC) &_bmms_bmms_vs, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_bmms(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
