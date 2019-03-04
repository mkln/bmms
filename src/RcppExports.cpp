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
// struct2d_prior_ratio
double struct2d_prior_ratio(const arma::field<arma::vec>& proposed, const arma::field<arma::vec>& original, int stage, int p, double param);
RcppExport SEXP _bmms_struct2d_prior_ratio(SEXP proposedSEXP, SEXP originalSEXP, SEXP stageSEXP, SEXP pSEXP, SEXP paramSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type proposed(proposedSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type original(originalSEXP);
    Rcpp::traits::input_parameter< int >::type stage(stageSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type param(paramSEXP);
    rcpp_result_gen = Rcpp::wrap(struct2d_prior_ratio(proposed, original, stage, p, param));
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
Rcpp::List sofk_binary(const arma::vec& y, const arma::mat& X, arma::field<arma::vec> start_splits, unsigned int mcmc, unsigned int burn, double lambda, int ii, int ll, bool silent, double structpar);
RcppExport SEXP _bmms_sofk_binary(SEXP ySEXP, SEXP XSEXP, SEXP start_splitsSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP lambdaSEXP, SEXP iiSEXP, SEXP llSEXP, SEXP silentSEXP, SEXP structparSEXP) {
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
    Rcpp::traits::input_parameter< double >::type structpar(structparSEXP);
    rcpp_result_gen = Rcpp::wrap(sofk_binary(y, X, start_splits, mcmc, burn, lambda, ii, ll, silent, structpar));
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
// div_by_colsum
arma::mat div_by_colsum(const arma::mat& J);
RcppExport SEXP _bmms_div_by_colsum(SEXP JSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type J(JSEXP);
    rcpp_result_gen = Rcpp::wrap(div_by_colsum(J));
    return rcpp_result_gen;
END_RCPP
}
// bdet
double bdet(const arma::mat& X);
RcppExport SEXP _bmms_bdet(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(bdet(X));
    return rcpp_result_gen;
END_RCPP
}
// totsplit_prior2_ratio
double totsplit_prior2_ratio(int tot_split_prop, int tot_split_orig, int norp, int ss, double lambda_prop);
RcppExport SEXP _bmms_totsplit_prior2_ratio(SEXP tot_split_propSEXP, SEXP tot_split_origSEXP, SEXP norpSEXP, SEXP ssSEXP, SEXP lambda_propSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type tot_split_prop(tot_split_propSEXP);
    Rcpp::traits::input_parameter< int >::type tot_split_orig(tot_split_origSEXP);
    Rcpp::traits::input_parameter< int >::type norp(norpSEXP);
    Rcpp::traits::input_parameter< int >::type ss(ssSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_prop(lambda_propSEXP);
    rcpp_result_gen = Rcpp::wrap(totsplit_prior2_ratio(tot_split_prop, tot_split_orig, norp, ss, lambda_prop));
    return rcpp_result_gen;
END_RCPP
}
// split_struct_ratio2
double split_struct_ratio2(const arma::field<arma::vec>& proposed, const arma::field<arma::vec>& original, int stage, int p, double param);
RcppExport SEXP _bmms_split_struct_ratio2(SEXP proposedSEXP, SEXP originalSEXP, SEXP stageSEXP, SEXP pSEXP, SEXP paramSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type proposed(proposedSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type original(originalSEXP);
    Rcpp::traits::input_parameter< int >::type stage(stageSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type param(paramSEXP);
    rcpp_result_gen = Rcpp::wrap(split_struct_ratio2(proposed, original, stage, p, param));
    return rcpp_result_gen;
END_RCPP
}
// wavelettize
arma::mat wavelettize(const arma::mat& J);
RcppExport SEXP _bmms_wavelettize(SEXP JSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type J(JSEXP);
    rcpp_result_gen = Rcpp::wrap(wavelettize(J));
    return rcpp_result_gen;
END_RCPP
}
// tline
double tline(const double& x, const double& m);
RcppExport SEXP _bmms_tline(SEXP xSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const double& >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(tline(x, m));
    return rcpp_result_gen;
END_RCPP
}
// Jcol_ilogitsmooth
arma::vec Jcol_ilogitsmooth(const arma::vec& J, double r);
RcppExport SEXP _bmms_Jcol_ilogitsmooth(SEXP JSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type J(JSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(Jcol_ilogitsmooth(J, r));
    return rcpp_result_gen;
END_RCPP
}
// Jcol_pnormsmooth
arma::vec Jcol_pnormsmooth(const arma::vec& J, double r);
RcppExport SEXP _bmms_Jcol_pnormsmooth(SEXP JSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type J(JSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(Jcol_pnormsmooth(J, r));
    return rcpp_result_gen;
END_RCPP
}
// J_smooth
arma::mat J_smooth(const arma::mat& J, double radius, bool nested);
RcppExport SEXP _bmms_J_smooth(SEXP JSEXP, SEXP radiusSEXP, SEXP nestedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type J(JSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< bool >::type nested(nestedSEXP);
    rcpp_result_gen = Rcpp::wrap(J_smooth(J, radius, nested));
    return rcpp_result_gen;
END_RCPP
}
// multi_split_nonnested
arma::mat multi_split_nonnested(const arma::mat& prevmat, arma::vec newsplits, int p);
RcppExport SEXP _bmms_multi_split_nonnested(SEXP prevmatSEXP, SEXP newsplitsSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type prevmat(prevmatSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type newsplits(newsplitsSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_split_nonnested(prevmat, newsplits, p));
    return rcpp_result_gen;
END_RCPP
}
// multi_split_new
arma::mat multi_split_new(const arma::vec& pones, const arma::vec& splits, int p);
RcppExport SEXP _bmms_multi_split_new(SEXP ponesSEXP, SEXP splitsSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type pones(ponesSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_split_new(pones, splits, p));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bmms_soi_cpp", (DL_FUNC) &_bmms_soi_cpp, 13},
    {"_bmms_soi_binary_cpp", (DL_FUNC) &_bmms_soi_binary_cpp, 14},
    {"_bmms_index_to_subscript", (DL_FUNC) &_bmms_index_to_subscript, 2},
    {"_bmms_load_splits", (DL_FUNC) &_bmms_load_splits, 2},
    {"_bmms_struct2d_prior_ratio", (DL_FUNC) &_bmms_struct2d_prior_ratio, 5},
    {"_bmms_sofk", (DL_FUNC) &_bmms_sofk, 15},
    {"_bmms_sofk_binary", (DL_FUNC) &_bmms_sofk_binary, 10},
    {"_bmms_bmms_base", (DL_FUNC) &_bmms_bmms_base, 8},
    {"_bmms_div_by_colsum", (DL_FUNC) &_bmms_div_by_colsum, 1},
    {"_bmms_bdet", (DL_FUNC) &_bmms_bdet, 1},
    {"_bmms_totsplit_prior2_ratio", (DL_FUNC) &_bmms_totsplit_prior2_ratio, 5},
    {"_bmms_split_struct_ratio2", (DL_FUNC) &_bmms_split_struct_ratio2, 5},
    {"_bmms_wavelettize", (DL_FUNC) &_bmms_wavelettize, 1},
    {"_bmms_tline", (DL_FUNC) &_bmms_tline, 2},
    {"_bmms_Jcol_ilogitsmooth", (DL_FUNC) &_bmms_Jcol_ilogitsmooth, 2},
    {"_bmms_Jcol_pnormsmooth", (DL_FUNC) &_bmms_Jcol_pnormsmooth, 2},
    {"_bmms_J_smooth", (DL_FUNC) &_bmms_J_smooth, 3},
    {"_bmms_multi_split_nonnested", (DL_FUNC) &_bmms_multi_split_nonnested, 3},
    {"_bmms_multi_split_new", (DL_FUNC) &_bmms_multi_split_new, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_bmms(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
