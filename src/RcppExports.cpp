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
// square_coord
arma::mat square_coord(const arma::vec& onesplit, int p1, int p2, int radius_int);
RcppExport SEXP _bmms_square_coord(SEXP onesplitSEXP, SEXP p1SEXP, SEXP p2SEXP, SEXP radius_intSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type onesplit(onesplitSEXP);
    Rcpp::traits::input_parameter< int >::type p1(p1SEXP);
    Rcpp::traits::input_parameter< int >::type p2(p2SEXP);
    Rcpp::traits::input_parameter< int >::type radius_int(radius_intSEXP);
    rcpp_result_gen = Rcpp::wrap(square_coord(onesplit, p1, p2, radius_int));
    return rcpp_result_gen;
END_RCPP
}
// split_move2d
arma::mat split_move2d(const arma::mat& mask_of_splits, const arma::mat& mask_nosplits, const arma::vec& onesplit, double& to_from_ratio, int radius_int);
RcppExport SEXP _bmms_split_move2d(SEXP mask_of_splitsSEXP, SEXP mask_nosplitsSEXP, SEXP onesplitSEXP, SEXP to_from_ratioSEXP, SEXP radius_intSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type mask_of_splits(mask_of_splitsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type mask_nosplits(mask_nosplitsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type onesplit(onesplitSEXP);
    Rcpp::traits::input_parameter< double& >::type to_from_ratio(to_from_ratioSEXP);
    Rcpp::traits::input_parameter< int >::type radius_int(radius_intSEXP);
    rcpp_result_gen = Rcpp::wrap(split_move2d(mask_of_splits, mask_nosplits, onesplit, to_from_ratio, radius_int));
    return rcpp_result_gen;
END_RCPP
}
// insert_empty_levels
arma::field<arma::mat> insert_empty_levels(const arma::field<arma::mat>& splitsub, const arma::vec& nctr_at_lev);
RcppExport SEXP _bmms_insert_empty_levels(SEXP splitsubSEXP, SEXP nctr_at_levSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type splitsub(splitsubSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nctr_at_lev(nctr_at_levSEXP);
    rcpp_result_gen = Rcpp::wrap(insert_empty_levels(splitsub, nctr_at_lev));
    return rcpp_result_gen;
END_RCPP
}
// soi_cpp
Rcpp::List soi_cpp(arma::vec y, arma::cube X, arma::field<arma::mat> centers, arma::mat mask_forbid, double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius, int start_movinglev, int partnum, bool save, bool save_more_data, bool fixsigma, arma::vec gin, bool try_bubbles);
RcppExport SEXP _bmms_soi_cpp(SEXP ySEXP, SEXP XSEXP, SEXP centersSEXP, SEXP mask_forbidSEXP, SEXP lambda_centersSEXP, SEXP lambda_ridgeSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP radiusSEXP, SEXP start_movinglevSEXP, SEXP partnumSEXP, SEXP saveSEXP, SEXP save_more_dataSEXP, SEXP fixsigmaSEXP, SEXP ginSEXP, SEXP try_bubblesSEXP) {
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
    Rcpp::traits::input_parameter< bool >::type save_more_data(save_more_dataSEXP);
    Rcpp::traits::input_parameter< bool >::type fixsigma(fixsigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gin(ginSEXP);
    Rcpp::traits::input_parameter< bool >::type try_bubbles(try_bubblesSEXP);
    rcpp_result_gen = Rcpp::wrap(soi_cpp(y, X, centers, mask_forbid, lambda_centers, lambda_ridge, mcmc, burn, radius, start_movinglev, partnum, save, save_more_data, fixsigma, gin, try_bubbles));
    return rcpp_result_gen;
END_RCPP
}
// soi_tester
Rcpp::List soi_tester(arma::vec y, arma::cube X, arma::field<arma::mat> centers, arma::field<arma::mat> to_centers, int levelchg, arma::mat mask_forbid, double sigmasq, double lambda_ridge, bool fixsigma, arma::vec gin, double bubbles_radius);
RcppExport SEXP _bmms_soi_tester(SEXP ySEXP, SEXP XSEXP, SEXP centersSEXP, SEXP to_centersSEXP, SEXP levelchgSEXP, SEXP mask_forbidSEXP, SEXP sigmasqSEXP, SEXP lambda_ridgeSEXP, SEXP fixsigmaSEXP, SEXP ginSEXP, SEXP bubbles_radiusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type centers(centersSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type to_centers(to_centersSEXP);
    Rcpp::traits::input_parameter< int >::type levelchg(levelchgSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mask_forbid(mask_forbidSEXP);
    Rcpp::traits::input_parameter< double >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_ridge(lambda_ridgeSEXP);
    Rcpp::traits::input_parameter< bool >::type fixsigma(fixsigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gin(ginSEXP);
    Rcpp::traits::input_parameter< double >::type bubbles_radius(bubbles_radiusSEXP);
    rcpp_result_gen = Rcpp::wrap(soi_tester(y, X, centers, to_centers, levelchg, mask_forbid, sigmasq, lambda_ridge, fixsigma, gin, bubbles_radius));
    return rcpp_result_gen;
END_RCPP
}
// mixed_binary_cpp
Rcpp::List mixed_binary_cpp(arma::vec y, arma::cube X, arma::mat X_g, arma::field<arma::mat> centers, arma::mat mask_forbid, double lambda_centers, double lambda_ridge, int mcmc, int burn, int radius, int start_movinglev, int partnum, bool save, bool save_more_data, bool fixsigma, double gin, double g_vs, double module_prior_par_vs);
RcppExport SEXP _bmms_mixed_binary_cpp(SEXP ySEXP, SEXP XSEXP, SEXP X_gSEXP, SEXP centersSEXP, SEXP mask_forbidSEXP, SEXP lambda_centersSEXP, SEXP lambda_ridgeSEXP, SEXP mcmcSEXP, SEXP burnSEXP, SEXP radiusSEXP, SEXP start_movinglevSEXP, SEXP partnumSEXP, SEXP saveSEXP, SEXP save_more_dataSEXP, SEXP fixsigmaSEXP, SEXP ginSEXP, SEXP g_vsSEXP, SEXP module_prior_par_vsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X_g(X_gSEXP);
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
    Rcpp::traits::input_parameter< bool >::type save_more_data(save_more_dataSEXP);
    Rcpp::traits::input_parameter< bool >::type fixsigma(fixsigmaSEXP);
    Rcpp::traits::input_parameter< double >::type gin(ginSEXP);
    Rcpp::traits::input_parameter< double >::type g_vs(g_vsSEXP);
    Rcpp::traits::input_parameter< double >::type module_prior_par_vs(module_prior_par_vsSEXP);
    rcpp_result_gen = Rcpp::wrap(mixed_binary_cpp(y, X, X_g, centers, mask_forbid, lambda_centers, lambda_ridge, mcmc, burn, radius, start_movinglev, partnum, save, save_more_data, fixsigma, gin, g_vs, module_prior_par_vs));
    return rcpp_result_gen;
END_RCPP
}
// reshape_mat
arma::vec reshape_mat(arma::mat X);
RcppExport SEXP _bmms_reshape_mat(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(reshape_mat(X));
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
// bmms_vs2
Rcpp::List bmms_vs2(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, const arma::field<arma::vec>& starting, int mcmc_in, arma::vec gg, arma::vec module_prior_par, bool binary);
RcppExport SEXP _bmms_bmms_vs2(SEXP y_inSEXP, SEXP Xall_inSEXP, SEXP startingSEXP, SEXP mcmc_inSEXP, SEXP ggSEXP, SEXP module_prior_parSEXP, SEXP binarySEXP) {
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
    rcpp_result_gen = Rcpp::wrap(bmms_vs2(y_in, Xall_in, starting, mcmc_in, gg, module_prior_par, binary));
    return rcpp_result_gen;
END_RCPP
}
// bmms_vs_tester
Rcpp::List bmms_vs_tester(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, const arma::field<arma::vec>& gamma, const arma::field<arma::vec>& gamma_alt, arma::vec gg, arma::vec module_prior_par);
RcppExport SEXP _bmms_bmms_vs_tester(SEXP y_inSEXP, SEXP Xall_inSEXP, SEXP gammaSEXP, SEXP gamma_altSEXP, SEXP ggSEXP, SEXP module_prior_parSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y_in(y_inSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type Xall_in(Xall_inSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type gamma_alt(gamma_altSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gg(ggSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type module_prior_par(module_prior_parSEXP);
    rcpp_result_gen = Rcpp::wrap(bmms_vs_tester(y_in, Xall_in, gamma, gamma_alt, gg, module_prior_par));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bmms_load_splits", (DL_FUNC) &_bmms_load_splits, 2},
    {"_bmms_square_coord", (DL_FUNC) &_bmms_square_coord, 4},
    {"_bmms_split_move2d", (DL_FUNC) &_bmms_split_move2d, 5},
    {"_bmms_insert_empty_levels", (DL_FUNC) &_bmms_insert_empty_levels, 2},
    {"_bmms_soi_cpp", (DL_FUNC) &_bmms_soi_cpp, 16},
    {"_bmms_soi_tester", (DL_FUNC) &_bmms_soi_tester, 11},
    {"_bmms_mixed_binary_cpp", (DL_FUNC) &_bmms_mixed_binary_cpp, 18},
    {"_bmms_reshape_mat", (DL_FUNC) &_bmms_reshape_mat, 1},
    {"_bmms_sofk", (DL_FUNC) &_bmms_sofk, 15},
    {"_bmms_sofk_binary", (DL_FUNC) &_bmms_sofk_binary, 15},
    {"_bmms_bmms_base", (DL_FUNC) &_bmms_bmms_base, 8},
    {"_bmms_bmms_debug", (DL_FUNC) &_bmms_bmms_debug, 8},
    {"_bmms_bmms_vs", (DL_FUNC) &_bmms_bmms_vs, 7},
    {"_bmms_bmms_vs2", (DL_FUNC) &_bmms_bmms_vs2, 7},
    {"_bmms_bmms_vs_tester", (DL_FUNC) &_bmms_bmms_vs_tester, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_bmms(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
