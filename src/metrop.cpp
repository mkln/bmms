//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

#include "metrop_helper.h"

using namespace std;

int MCMCSWITCH = 0;

arma::vec proposal_jumplr_rj(const arma::field<arma::vec>& current_splits, int stage, int split, int p, double decay, bool nested=true){
  //cout << "[][][][][][][][] proposal move rj [][][][][][][][]" << endl;
  arma::vec all_splits = arma::zeros(0);
  if(!nested){ 
    for(unsigned int s=0; s<current_splits.n_elem; s++){
      if(current_splits(s).n_elem > 0){
        all_splits = arma::join_vert(all_splits, current_splits(s));
      } else {
        break;
      }
    }
  } else { // allow overlaps across levels
    all_splits = current_splits(stage);
  }
  //cout << current_splits << endl << "[ " << stage << " " << split <<  " ] " << endl;
  int starting_from = current_splits(stage)(split);
  
  // determining move
  int move;
  if((p > 500) & false){
    try
    {
      // will select in 0:p-2
      move = bmrandom::rndpp_sample1_comp(all_splits, p-2, starting_from, stage, current_splits.n_elem);
    } catch (...){
      clog << "Warning: issue encountered when moving, running alternative proposal." << endl;
      move = bmrandom::rndpp_sample1_comp_alt(all_splits, p-2, starting_from, .1);
    }
  } else {
    move = bmrandom::rndpp_sample1_comp_alt(all_splits, p-2, starting_from, .2);
  }
  
  double ip_move_forward = p-1 - all_splits.n_elem ; // = 1/q(new | old)
  double ip_move_backward = p-1 - all_splits.n_elem; // = 1/(q(old | new))
  
  // rj ratio is q(old | new) / q(new | old) = ip_move_forward/ip_move_backward
  
  arma::vec results = arma::zeros(2);
  results(0) = move;
  results(1) = ip_move_forward / ip_move_backward;
  cout << "move split? " << ip_move_forward << " <> " << ip_move_backward << " :: " << results(1) << endl;
  return results;
}


arma::field<arma::vec> proposal_move_split(const arma::field<arma::vec>& current_splits, 
                                           int stage, int split, 
                                           const arma::vec& y, 
                                           const arma::mat& X, int p, int n,
                                           ModularLinReg& base_model, double decay){
  // initialize the proposal as the last accepted value of the splits
  arma::field<arma::vec> proposed_splits = current_splits;
  
  // proposing a change of place of the split
  // same probability of moving back here if we were in the proposal
  
  arma::vec proposal_rj = proposal_jumplr_rj(current_splits, stage, split, p, decay, base_model.nested);
  int move_instage = proposal_rj(0); //bmrandom::rndpp_discrete({1, 1})*2-1; // -1: move-, 1: move+ 
  double rj_prob = proposal_rj(1);
  
  double ssratio;
  //cout << ">> from " << proposed_splits << " to " << proposed_splits(stage)(split)+move_instage << endl;
  //clog << ">> from " << proposed_splits(stage)(split) << " to " << move_instage << "," << " in " << stage << endl;
  
  if(move_instage == -1){ // (!move_possible(proposed_splits, proposed_splits(stage)(split)+move_instage, stage, split, p)){
    // proposing a split at variable -1 or variable p, which is meaningless
    cout << "                      can't move here" << endl;
    return current_splits;
  } else {
    //proposed_splits(stage)(split) += move_instage;
    proposed_splits(stage)(split) = move_instage; // from jumplr
    //clog << "\n --> moving " << proposed_splits(stage)(split) << " from " << current_splits(stage)(split) << endl;
    
    
    ModularLinReg proposed_model = base_model;
    if(stage>0){
      proposed_model.change_module(stage, proposed_splits(stage)); //  = ModularLinReg(y, X, proposed_splits, base_model.max_stages);//
    } else {
      proposed_model = ModularLinReg(y, X, base_model.g_prior, proposed_splits, base_model.rad, base_model.max_stages, 
                                     base_model.fixed_sigma_v, base_model.fixed_splits,
                                     base_model.a, base_model.b, base_model.structpar);
    }

    //clog << "------------ move" << endl; 
    // comparison after a move can be done comparing marginal likelihoods
    ssratio = split_struct_ratio2(proposed_model.bigsplit, 
                                         base_model.bigsplit, stage, base_model.p, base_model.structpar);
    if(!isfinite(ssratio)){
      clog << "problem with moving. here's original" << endl;
      clog << base_model.bigsplit << endl;
      clog << "and split seq" << endl;
      clog << base_model.split_seq << endl;
      throw 1;
    }
    cout << proposed_model.loglik.t() << endl << base_model.loglik.t() << endl;
    
    double prob = false ? exp(proposed_model.loglik(stage) - base_model.loglik(stage)) : exp(arma::accu(proposed_model.loglik - base_model.loglik));
    prob = prob * rj_prob * ssratio;
    
/*
    clog << "mlik ratio of move: " <<  prob << endl;
    
    clog << "stage: " << stage <<  ". mlr move " << prob << " " << rj_prob << endl <<
      ">" << proposed_model.loglik.t() <<  
      ">" << base_model.loglik.t() << endl;
    clog << proposed_model.loglik(stage) - base_model.loglik(stage) << endl;
    
    clog << arma::join_horiz(proposed_model.modules[0].mean_post, base_model.modules[0].mean_post) << endl;
    */
      //  arma::accu(proposed_model.loglik) << " " << arma::accu(base_model.loglik) << " : " << 
      //    arma::accu(proposed_model.loglik - base_model.loglik) << endl;
    //clog << "here: " << proposed_model.loglik(stage) << " vs " << base_model.loglik(stage) << endl;
    
    prob = prob > 1 ? 1 : prob;
    int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
    if(accepted_proposal == 1){
      //clog << ssratio << endl;
      //clog << "[MOVE SPLIT " << stage << "] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
      base_model = proposed_model;
      //clog << "accepted" << endl;
      return proposed_splits;
      //cout << "         accepted: s: " << s << " j: " << j << " val: "<< splits(m)(s)(j) << endl;
    } else {
      //clog << "rejected" << endl;
      //clog << "[MOVE SPLIT " << stage << "] reject, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
      return current_splits;
      //cout << "         rejected: s: " << s << " j: " << j << " val: "<< splits(m)(s)(j) << endl;
    }
  }
}

arma::vec proposal_drop_rj(const arma::field<arma::vec>& current_splits, 
                           int stage, int p, int n, double lambda_prop=10.0,
                           bool nested=true){//, int stage){
  
  //cout << "[][][][][][][][] proposal move rj [][][][][][][][]" << endl;
  int all_splits_elem = 0;//arma::zeros(0);
  if(!nested){
    for(unsigned int s=0; s<current_splits.n_elem; s++){
      all_splits_elem += current_splits(s).n_elem;
    }
  } else { // allow overlaps
    all_splits_elem = current_splits(stage).n_elem;
  }
  
  // determining move
  int move = bmrandom::rndpp_unif_int(current_splits(stage).n_elem-1); // pick at random from the available at stage
  double ip_move_forward = current_splits(stage).n_elem; // = 1/q(new | old) // how many are available
  // backward move would be to add at this stage
  // we can add here picking at random from the available
  // total is p-1 splits, already busy is all_splits.n_elem-1 
  // available = p-1-all_splits.n_elem-1
  double ip_move_backward = p - all_splits_elem; // = 1/(q(old | new))
  
  // rj ratio is q(old | new) / q(new | old) = ip_move_forward/ip_move_backward
  arma::vec results = arma::zeros(2);
  results(0) = move;
  results(1) = ip_move_forward / ip_move_backward * totsplit_prior2_ratio(current_splits(stage).n_elem - 1, 
          current_splits(stage).n_elem, p<n?p:n, 
                                          current_splits.n_elem - stage - 1, 
                                          lambda_prop);
  //ip_move_forward / ip_move_backward * totsplit_prior_ratio(all_splits_elem - 1, all_splits_elem, p<n?p:n, stage);
  //cout << "drop split? " << ip_move_forward << " <> " << ip_move_backward << " :: " << results(1) << endl;
  //cout << "all_splits " << all_splits_elem << " p:" << p << " ?:" << p -all_splits_elem << endl;
  return results;
}


arma::field<arma::vec> proposal_drop_split(const arma::field<arma::vec>& current_splits, 
                                           int stage,  
                                           arma::vec& y, arma::mat& X, int p, int n,
                                           ModularLinReg& base_model,
                                           double lambda_prop=10.0){
  // removing something
  // allowed if
  // - proposed a drop
  // AND - we are at the last stage, OR at any stage if the number of splits is >1
  int all_splits_elem = 0;//arma::zeros(0);
  for(unsigned int s=0; s<current_splits.n_elem; s++){
    all_splits_elem += current_splits(s).n_elem;
  }
  
  arma::field<arma::vec> proposed_splits = current_splits;
  
  arma::vec dropmove = proposal_drop_rj(current_splits, stage, p, n, lambda_prop, base_model.nested); //base_model.n_stages);
  int dropping = dropmove(0); //
  //int dropping = bmrandom::rndpp_unif_int(proposed_splits(stage).n_elem-1);
  double rj_prob = dropmove(1);
  //clog << "dropping with proposal prob " <<  rj_prob << endl;
  
  //cout << "         proposal = drop " << proposed_splits(stage)(dropping) << endl;
  proposed_splits(stage)(dropping) = -1;
  
  //cout << "fixing possible wrong splits " << endl;
  proposed_splits(stage) = bmfuncs::split_fix(proposed_splits, stage); // fix this but dont touch the stage
  //cout << "proposed splits with this drop proposals look like this: " << endl;
  //cout << proposed_splits << endl;
  //cout << "if I try fixing the stages then " << endl;
  //cout << stage_fix(proposed_splits) << endl;
  
  ModularLinReg proposed_model = base_model;
  
  double mlr;
  double prob, ssratio;
  
  if(current_splits(stage).n_elem == 1){
    // I'm proposing a drop of the last stage;
    proposed_model.delete_last_module();
    double dropping_stage = totstage_prior_ratio(base_model.n_stages - 1, base_model.n_stages, p<n?p:n, all_splits_elem, -1);
    // the proposed model is the same as the base model, except for the last stage
    // we integrate from the base, we fix at the proposed
    mlr = exp(proposed_model.loglik(proposed_model.n_stages-1) - base_model.loglik(base_model.n_stages-1));
    prob = mlr * dropping_stage;
  } else {
    // drop of a split at some stage amounts to a change of that stage
    if(stage>0){
      proposed_model.change_module(stage, proposed_splits(stage)); // = ModularLinReg(y, X, proposed_splits, base_model.max_stages); //
    } else {
      proposed_model = ModularLinReg(y, X, base_model.g_prior, proposed_splits, 
                                     base_model.rad, base_model.max_stages, 
                                     base_model.fixed_sigma_v, base_model.fixed_splits,
                                     base_model.a, base_model.b, base_model.structpar);
    }
    ssratio = split_struct_ratio2(proposed_model.bigsplit, 
                                  base_model.bigsplit, stage, base_model.p, base_model.structpar);
    
    prob = false ? exp(proposed_model.loglik(stage) - base_model.loglik(stage)) : exp(arma::accu(proposed_model.loglik - base_model.loglik));
    prob = prob * rj_prob * ssratio;
  }
  
  prob = prob > 1 ? 1 : prob;
  int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
  if(accepted_proposal == 1){
    cout << "[DROP SPLIT] accept, MLR: " << mlr;
    cout << (current_splits(stage).n_elem == 1 ? " <STAGE> " : "" );
    cout << endl;
    
    base_model = proposed_model;
    return proposed_splits; // save with fix
  } else {
    cout << "[DROP SPLIT] reject, MLR: " << mlr << " ssratio=" << ssratio << endl;
    cout << (current_splits(stage).n_elem == 1 ? " <STAGE> " : "" );
    cout << endl;
    return current_splits;
  }
}


arma::field<arma::vec> proposal_drop_split_2(const arma::field<arma::vec>& current_splits, 
                                             int stage,  
                                             const arma::vec& y, 
                                             const arma::mat& X, int p, int n,
                                             ModularLinReg& base_model,
                                             double lambda_prop=10.0,
                                             bool mnpr=false){
  // removing something
  // allowed if
  // - proposed a drop
  // AND - we are at the last stage, OR at any stage if the number of splits is >1
  int all_splits_elem = 0;//arma::zeros(0);
  for(unsigned int s=0; s<current_splits.n_elem; s++){
    all_splits_elem += current_splits(s).n_elem;
  }
  
  arma::field<arma::vec> proposed_splits = current_splits;
  
  arma::vec dropmove = proposal_drop_rj(current_splits, stage, p, n, lambda_prop, base_model.nested); //base_model.n_stages);
  int dropping = dropmove(0); //
  //int dropping = bmrandom::rndpp_unif_int(proposed_splits(stage).n_elem-1);
  double rj_prob = dropmove(1);
  //clog << "dropping with proposal prob " <<  rj_prob << endl;
  
  //cout << "         proposal = drop " << proposed_splits(stage)(dropping) << endl;
  proposed_splits(stage)(dropping) = -1;
  
  //cout << "fixing possible wrong splits " << endl;
  proposed_splits(stage) = bmfuncs::split_fix(proposed_splits, stage); // fix this but dont touch the stage
  //cout << "proposed splits with this drop proposals look like this: " << endl;
  //cout << proposed_splits << endl;
  //cout << "if I try fixing the stages then " << endl;
  //cout << stage_fix(proposed_splits) << endl;
  
  ModularLinReg proposed_model = base_model;
  
  double mlr; 
  double ssratio;
  double prob;
  
  if(current_splits(stage).n_elem == 1){
    prob = 0.0;
  } else {
    rj_prob = rj_prob ;
    // drop of a split at some stage amounts to a change of that stage
    if(stage>0){
      proposed_model.change_module(stage, proposed_splits(stage)); // = ModularLinReg(y, X, proposed_splits, base_model.max_stages); //
    } else {
      proposed_model = ModularLinReg(y, X, base_model.g_prior, proposed_splits, 
                                     base_model.rad, base_model.max_stages, 
                                     base_model.fixed_sigma_v, base_model.fixed_splits,
                                     base_model.a, base_model.b, base_model.structpar);
    }
    prob = false ? exp(proposed_model.loglik(stage) - base_model.loglik(stage)) : exp(arma::accu(proposed_model.loglik - base_model.loglik));
    prob = prob * rj_prob * ssratio;
    
    //clog << "------------ drop" << endl; 
    ssratio = split_struct_ratio2(proposed_model.bigsplit, 
                                         base_model.bigsplit, stage, base_model.p, base_model.structpar);
    prob = prob * rj_prob * ssratio;

    //clog << proposed_model.loglik << " " << base_model.loglik << endl;
    //clog << mlr << " " << rj_prob << endl;
  }
  
  prob = prob > 1 ? 1 : prob;
  int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
  if(accepted_proposal == 1){
    //clog << ssratio << endl;
    cout << "[DROP SPLIT] accept, MLR: " << mlr;
    cout << (current_splits(stage).n_elem == 1 ? " <STAGE> " : "" );
    cout << endl;
    
    base_model = proposed_model;
    return proposed_splits; // save with fix
  } else {
    cout << "[DROP SPLIT] reject, MLR: " << mlr << " ssratio=" << ssratio << endl;
    cout << (current_splits(stage).n_elem == 1 ? " <STAGE> " : "" );
    cout << endl;
    return current_splits;
  }
}



arma::field<arma::vec> proposal_drop_stage(const arma::field<arma::vec>& current_splits, 
                                           ModularLinReg& base_model){
  // removing something
  // allowed if
  // - proposed a drop
  // AND - we are at the last stage, OR at any stage if the number of splits is >1
  int all_splits_elem = 0;//arma::zeros(0);
  for(unsigned int s=0; s<current_splits.n_elem; s++){
    all_splits_elem += current_splits(s).n_elem;
  }
  
  arma::field<arma::vec> proposed_splits = current_splits;
  proposed_splits(base_model.n_stages-1) = arma::zeros(0);
  
  //arma::vec dropmove = proposal_drop_rj(current_splits, stage, p, n, base_model.n_stages);
  //int dropping = dropmove(0); //
  //int dropping = bmrandom::rndpp_unif_int(proposed_splits(stage).n_elem-1);
  double rj_prob = 1.0;//dropmove(1);
  //clog << "dropping with proposal prob " <<  rj_prob << endl;
  
  //cout << "         proposal = drop " << proposed_splits(stage)(dropping) << endl;
  //proposed_splits(stage)(dropping) = -1;
  
  //cout << "fixing possible wrong splits " << endl;
  //proposed_splits(stage) = bmfuncs::split_fix(proposed_splits, stage); // fix this but dont touch the stage
  //cout << "proposed splits with this drop proposals look like this: " << endl;
  //cout << proposed_splits << endl;
  
  cout << "if I try fixing the stages then " << endl;
  cout << bmfuncs::stage_fix(proposed_splits) << endl;
  cout << "done" << endl;
  
  ModularLinReg proposed_model = base_model;
  
  double mlr;
  double prob;
  
  int n=base_model.n;
  int p=base_model.p;
  
  proposed_model.delete_last_module();
  int prop_stages = proposed_model.n_stages;
  cout << "boh?" << endl;
  double dropping_stage = totstage_prior_ratio(base_model.n_stages - 1, base_model.n_stages, p<n?p:n, all_splits_elem, -1);
  // the proposed model is the same as the base model, except for the last stage
  // we integrate from the base, we fix at the proposed
  //exp(proposed_model.loglik(proposed_model.n_stages-1) - base_model.loglik(base_model.n_stages-1));
  
  Module prop_last = proposed_model.modules[base_model.n_stages-1];
  prob = exp(modular_loglik0(proposed_model.modules[prop_stages-1].ej, proposed_model.modules[prop_stages-1].a, proposed_model.modules[prop_stages-1].b)
    - base_model.loglik(proposed_model.n_stages-1));
    
    //exp(modular_loglikn(prop_last.ej - prop_last.xb_mean, 1.0/prop_last.sigmasq_sample * base_model.In) 
         //             - base_model.loglik(base_model.n_stages-2)) * 
    //rj_prob *
    //ssratio;

  prob = prob > 1 ? 1 : prob;
  int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
  if(accepted_proposal == 1){
    //clog << "-";
    //clog << "[DROP STAGE] accept, MLR: " << mlr << endl;
    
    base_model = proposed_model;
    return proposed_splits; // save with fix
  } else {
    //clog << "[DROP STAGE] reject, MLR: " << mlr << endl;
    return current_splits;
  }
}


arma::vec proposal_add_rj(const arma::field<arma::vec>& current_splits, 
                          int stage, int p, int n, double lambda_prop=10.0,
                          bool nested=true){ //, int stage){
  arma::vec all_splits = arma::zeros(0);
  if(!nested){
    for(unsigned int s=0; s<current_splits.n_elem; s++){
      if(current_splits(s).n_elem > 0){
        all_splits = arma::join_vert(all_splits, current_splits(s));
      } else {
        break;
      }
    }
  } else { // allow overlaps
    all_splits = current_splits(stage);
  }
  
  
  // determining move
  //clog << "maybe here 1" << endl;
  int move = bmrandom::rndpp_sample1_comp(all_splits, p-2, -1, stage+1, current_splits.n_elem);
  //int move = bmrandom::rndpp_sample1_comp(all_splits, p-2, -1, 1.0, 0);  // pick at random from the available from p-1
  //clog << "maybe here 2 " << endl;
  double ip_move_forward = p-1 - all_splits.n_elem; // = 1/q(new | old) // how many are available
  // backward move would be to drop at this stage
  // we can drop here picking at random from the available at this stage
  // available is current_splits(stage).n_elem+1 splits
  double ip_move_backward = 1+current_splits(stage).n_elem; // = 1/(q(old | new))
  
  // rj ratio is q(old | new) / q(new | old) = ip_move_forward/ip_move_backward
  arma::vec results = arma::zeros(2);
  results(0) = move;
  //clog << "prior ratio adding " << totsplit_prior_ratio(all_splits.n_elem + 1, all_splits.n_elem, p, stage) << endl;
  results(1) = ip_move_forward / ip_move_backward * 
    totsplit_prior2_ratio(current_splits(stage).n_elem + 1, 
                          current_splits(stage).n_elem, p<n?p:n, 
                                                          current_splits.n_elem - stage - 1, 
                                                          lambda_prop);//ip_move_forward / ip_move_backward * totsplit_prior_ratio(all_splits.n_elem + 1, all_splits.n_elem, p<n?p:n, stage);

  return results;
}

arma::field<arma::vec> proposal_add_split(const arma::field<arma::vec>& current_splits, 
                                          int stage,  
                                          const arma::vec& y, 
                                          const arma::mat& X, int p, int n,
                                          ModularLinReg& base_model,
                                          double lambda_prop=10.0,
                                          bool mnpr=false){
  // probability of going from base to proposal (add)
  // is 0.5 * 1/(p-elem)
  arma::vec addsplit_rj = proposal_add_rj(current_splits, stage, p, n, lambda_prop, base_model.nested);//base_model.n_stages);
  int new_split = addsplit_rj(0);
  double rj_prob = addsplit_rj(1);
  //clog << "adding with proposal prob " <<  rj_prob << endl;
  
  //bmrandom::rndpp_sample1_comp(current_splits(stage), p-1); //can't sample the last one because it's not a split
  double ssratio;
  //clog << "proposal: add " << new_split << " to stage " << stage << " ... ";
  //move_possible(current_splits, new_split, stage, current_splits(stage).n_elem, p)
  if((new_split > -1)){
    arma::field<arma::vec> proposed_splits = current_splits;
    proposed_splits(stage) = arma::vec(current_splits(stage).n_elem + 1);
    proposed_splits(stage).subvec(0, current_splits(stage).n_elem-1 ) = current_splits(stage);
    //cout << "         proposal = add " << new_split << endl;
    
    proposed_splits(stage)(current_splits(stage).n_elem) = new_split;
    
    //cout << "fixing possible wrong splits " << endl;
    ModularLinReg proposed_model = base_model;
    if(stage>0){
      proposed_model.change_module(stage, proposed_splits(stage)); //  = ModularLinReg(y, X, proposed_splits, base_model.max_stages);
    } else {
      //clog << base_model.g_prior << endl <<
      //  base_model.kernel_type << endl <<
      //  base_model.max_stages << endl <<
      //  base_model.fix_sigma << endl <<
      //  base_model.fixed_splits;
      proposed_model = ModularLinReg(y, X, base_model.g_prior, proposed_splits, 
                                     base_model.rad, base_model.max_stages, 
                                     base_model.fixed_sigma_v, base_model.fixed_splits,
                                     base_model.a, base_model.b, base_model.structpar);
    }
    //ModularLinReg proposed_model = base_model;
    //proposed_model.rebuild(proposed_splits);
    
    // same stages
    //clog << "------------ add" << endl; 
    ssratio = split_struct_ratio2(proposed_model.bigsplit, 
                                         base_model.bigsplit, stage, base_model.p, base_model.structpar);
    
    double prob = false ? exp(proposed_model.loglik(stage) - base_model.loglik(stage)) : exp(arma::accu(proposed_model.loglik - base_model.loglik));
    prob = prob * rj_prob * ssratio;

    
    prob = prob > 1 ? 1 : prob;
    double accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
    if(accepted_proposal == 1){
      //clog << ssratio << endl;
      cout << "[ADD SPLIT] accept, MLR: " << exp(arma::accu(proposed_model.loglik) - arma::accu(base_model.loglik)) << " rjprob=" << rj_prob << " ssratio=" << ssratio << endl;
      base_model = proposed_model;
      return proposed_splits;
      
    } else {
      cout << "[ADD SPLIT] reject, MLR: " << exp(arma::accu(proposed_model.loglik) - arma::accu(base_model.loglik)) << " rjprob=" << rj_prob << " ssratio=" << ssratio << endl;
      return current_splits;
    }
  } else {
    //cout << "not allowed." << endl;
    return current_splits;
  }
}

arma::vec proposal_newstage_rj(const arma::field<arma::vec>& current_splits, int n, int p, int curr_n_stages){
  
  //cout << "[][][][][][][][] proposal move rj [][][][][][][][]" << endl;
  arma::vec all_splits = arma::zeros(0);
  for(unsigned int s=0; s<current_splits.n_elem; s++){
    all_splits = arma::join_vert(all_splits, current_splits(s));
  }
  // determining move
  int move = bmrandom::rndpp_sample1_comp(all_splits, p-2, -1, 1.0, 0);  // pick at random from the available from p-2
  // p-1 is the last variable and cant split there
  
  double ip_move_forward = p-1 - all_splits.n_elem; // = 1/q(new | old) // how many are available
  // backward move would be to drop at this stage
  // we can drop here picking at random from the available at this stage
  // available is current_splits(stage).n_elem+1 splits
  double ip_move_backward = 1; // = 1/(q(old | new))
  
  // rj ratio is q(old | new) / q(new | old) = ip_move_forward/ip_move_backward
  arma::vec results = arma::zeros(2);
  results(0) = move;
  //(int tot_stage_prop, int tot_stage_orig, int p, int curr_n_splits)
  //clog << totstage_prior_ratio(curr_n_stages + 1, curr_n_stages, p, all_splits.n_elem, 1) <<" <-- newstage prior ratio" << endl;
  results(1) = totstage_prior_ratio(curr_n_stages + 1, curr_n_stages, p<n?p:n, all_splits.n_elem, 1) * ip_move_forward / ip_move_backward ;
  //exp(- pow(all_splits.n_elem+1 - p/3.0, 2) + pow(all_splits.n_elem - p/3.0, 2) ); // * 
  //exp( - 100* pow(current_splits.n_elem+1 - log2(p+0.0), 2) + 100* pow(current_splits.n_elem - log2(p+0.0), 2) );
  cout << "new stage? " << ip_move_forward << " <> " << ip_move_backward << " :: " << results(1) << endl;
  return results;
}

arma::field<arma::vec> proposal_add_stage(const arma::field<arma::vec>& current_splits, 
                                          const arma::vec& y, const arma::mat& X, int p, int n,
                                          ModularLinReg& base_model, int max_levels){
  //clog << "  !! trying to add new stage !! " << endl;

  
  int base_stages = base_model.n_stages;
  
  if(base_model.n_stages < max_levels){
    arma::vec addsplit_rj = proposal_newstage_rj(current_splits, n, p, base_model.n_stages);
    int split_at_new_stage = addsplit_rj(0);
    double rj_prob = addsplit_rj(1);
    
    arma::field<arma::vec> proposed_splits = current_splits;
    
    // LIFO
    for(unsigned int s=0; s< max_levels; s++){
      if(current_splits(s).n_elem == 0){
        proposed_splits(s) = split_at_new_stage;
        break;
      }
    } 
    
    if(split_at_new_stage > -1){ //move_possible(proposed_splits, split_at_new_stage, proposed_splits.n_elem-1, 0, p)
      //cout << "adding a split at " << split_at_new_stage << endl;
      
      //cout << "the proposed splits are now" << endl << proposed_splits << endl; 
      ModularLinReg proposed_model = base_model; //(base_model.y, base_model.X, proposed_splits, max_levels);
      proposed_model.add_new_module(proposed_splits(base_model.n_stages));
      cout << "proposal model for NEW STAGE built" << endl;
      
      double ssratio = split_struct_ratio2(proposed_model.bigsplit, 
                                    base_model.bigsplit, -1, base_model.p, base_model.structpar);
      
      // compare with empty model
      Module base_last = base_model.modules[base_model.n_stages-1];
      double prob = exp(proposed_model.loglik(proposed_model.n_stages-1) - 
                        modular_loglik0(base_model.modules[base_stages-1].ej, base_model.modules[base_stages-1].a, base_model.modules[base_stages-1].b)) *
                        rj_prob * ssratio;
        
        //exp(proposed_model.loglik(proposed_model.n_stages-1) - 
                    //    modular_loglikn(base_last.ej - base_last.xb_mean, 1.0/base_last.sigmasq_sample * base_model.In)
                    //    ) * 
                    //    rj_prob *
                     //   ssratio;
      //cout << "adding? MLR " << exp(proposed_model.loglik - base_model.loglik) << endl;
      prob = prob > 1 ? 1 : prob;
      int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
      if(accepted_proposal == 1){
        //clog << "+";
        //clog << "[ADD STAGE] accept, MLR: " << exp(proposed_model.loglik - base_model.loglik) << " now:" << proposed_model.n_stages << endl;
        /*
         arma::field<arma::vec> old_splitm = splits(m);
         splits(m) = arma::field<arma::vec>(proposed_splits.n_elem);
         
         for(unsigned int s=0; s<splits(m).n_elem-1; s++){
         splits(m)(s) = old_splitm(s);
         }
         splits(m)(finS) = proposed_splits(finS);
         */
        //cout << "         accepted: s: " << finS << endl;
        base_model = proposed_model;
        return proposed_splits;
      } else {
        //clog << "[ADD STAGE] reject, MLR: " << exp(proposed_model.loglik - base_model.loglik) << endl;
        //clog << " " << proposed_model.loglik << " " << base_model.loglik << endl;
        return current_splits;
      }
    } else {
      return current_splits;
    }
  } else {
    // no more levels
    return current_splits;
  }
} 

//' @export
// [[Rcpp::export]]
Rcpp::List sofk(const arma::vec& yin, const arma::mat& X, 
                const arma::field<arma::vec>& start_splits, 
                unsigned int mcmc = 100, unsigned int burn = 50,
                double lambda=5.0,
                double ain=2.1, double bin=1.1,
                int ii=0, int ll=0,
                bool onesigma = true,
                bool silent = true,
                double gin=0.01,
                double structpar=1.0,
                bool trysmooth=false){
  if(silent){ cout.setstate(std::ios_base::failbit); } else {  cout.clear(); }
  // sample from posterior of changepoints given their number
  // idea:
  // each stage will use increasing number of changepoints
  // posterior from first used as offset
  
  //double icept = arma::mean(yin);
  //arma::vec xs = bmdataman::col_sums(X.t());
  //arma::vec y = yin-icept;
  //double bmean = arma::conv_to<double>::from(arma::inv_sympd(xs.t()*xs)*xs.t()*y);
  arma::vec y = yin; //y-xs*bmean;
  
  int n = y.n_elem;
  int p = X.n_cols;
  int max_stages = start_splits.n_elem;
  
  double g=gin;
  
  MCMCSWITCH = 1;
  int choices = 3; //onesigma? 3 : 4;
  
  double icept;
  arma::mat theta_mcmc = arma::zeros(mcmc - burn, p);
  //arma::mat mu_mcmc = arma::zeros(mcmc - burn, p);
  arma::vec icept_mcmc = arma::zeros(mcmc-burn);
  arma::field<arma::field<arma::vec>> theta_ms = arma::field<arma::field<arma::vec>>(mcmc - burn);
  arma::field<arma::field<arma::vec>> mu_ms = arma::field<arma::field<arma::vec>>(mcmc - burn);
  
  arma::field<arma::field<arma::vec>> splits = arma::field<arma::field<arma::vec>>(mcmc);
  arma::field<arma::field<arma::vec>> splits_save = arma::field<arma::field<arma::vec>>(mcmc - burn);
  
  arma::mat sigmasq_ms = arma::zeros(mcmc - burn, max_stages);
  arma::vec sigmasq_mcmc = arma::vec(mcmc - burn);
  arma::vec radius_mcmc = arma::vec(mcmc-burn);
  //arma::mat pred_mcmc = arma::zeros(mcmc, n);
  
  // initial field of splits is a simple split at 2
  splits(0) = arma::field<arma::vec>(max_stages);
  splits(0) = start_splits;//sample_no_replace(p-2, numsplits);
  //clog << splits(0)(0) << endl;
  //splits(0)(1) = arma::ones(1)*(p-3); 
  
  arma::field<arma::vec> current_splits = arma::field<arma::vec>(max_stages);
  //arma::field<arma::vec> proposed_splits = arma::field<arma::vec>(max_stages);
  
  int more_or_less = 0; // 0=move, 1=add/del
  int which_mol = 0; // index used to remove or add split at a stage
  
  int move_instage = 0; // -1=move to left, 1=move to right
  int move_outstage = 0; // 0=stay, 1=add a split to next stage
  
  double prob=0;
  int accepted_proposal;
  
  double proposed = 0;
  double accepted = 0;
  
  cout << "first model" << endl;
  
  double radius = 0.0;
  if(trysmooth){
    radius = .5;
  }
  
  
  ModularLinReg base_model(y, X, g, splits(0), radius, max_stages, onesigma? 1.0 : -1.0, false, ain, bin, structpar);
  int m=0;
  
  arma::mat In = arma::eye(n,n);
  cout << "first model done " << endl;
  int tot_added_splits = 0;
  int tot_dropped_splits = 0;
  int tot_moved = 0;
  int tot_added_stages = 0;
  int tot_dropped_stages = 0;
  double decay = 2.0;
  double sigmasq_sample = 1.0;
  
  for(unsigned int m = 1; m < mcmc; m++){
    //clog << m << endl;
    Rcpp::checkUserInterrupt();
    cout << "========================================" << endl;
    //cout << "starting from" << endl;
    //cout << splits(m-1) << endl;
    //cout << "----------------------------------------" << endl;
    cout << "MCMC : " << m << endl;
    // start by copying last splits. we will change it if proposals are accepted
    splits(m) = splits(m-1);
    
    int n_stages = base_model.n_stages;
    //clog << n_stages << endl;
    
    // mean
    
    // move, add split, drop split, add stage, drop stage
    // this function only allows move
    //clog << "here 1" << endl;
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));
    //clog << "here 2 " << endl;
    
    if(move_type == 0){    // cycle through the stages. for each stage we go through the splits
      cout << "MOVING [" << m << "]" << endl;
      for(int s=n_stages-1; s>-1; s--){
        cout << "  STAGE : " << s << endl;
        // cycle through the splits. either change +1 or -1, or delete
        // for every proposal, choose whether to accept it
        // if accepted, save to splits at this mcmc stage
        for(unsigned int j=0; j<splits(m)(s).n_elem; j++){
          cout << "    SPLIT : " << j << endl;
          int curr_split = splits(m)(s)(j);
          //arma::vec diffs_pre = arma::diff(arma::sort(base_model.bigsplit(base_model.n_stages-1)));
          //arma::field<arma::vec> splitspre = base_model.bigsplit;
          arma::field<arma::vec> tempsplits = splits(m);
          try{
            tempsplits = proposal_move_split(splits(m), s, j, y, X, p, n, base_model, decay);
            splits(m) = tempsplits;
          } catch(...){
            clog << "Warning, got error in MCMC [moving] -- staying at old splits." << endl;
            splits(m) = tempsplits;
          }
          /*arma::vec diffs_post = arma::diff(arma::sort(base_model.bigsplit(base_model.n_stages-1)));
          if(diffs_post.min() == 0){
            clog << "originally" << endl << splitspre << endl << "then splitseq: " << endl;
            clog << base_model.split_seq << endl;
            clog << base_model.bigsplit << endl;
            clog << "n stages " << base_model.n_stages << endl;
            throw 1;
          }*/
          proposed++;
          if(curr_split != splits(m)(s)(j)){
            tot_moved ++;
            accepted++;
          }
        } // split loop
      }
      //cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }
    if(move_type == 1){
      cout << "ADD SPLIT" << endl;
      for(int s=n_stages-1; s>-1; s--){
        cout << "  STAGE : " << s << endl;
        unsigned int curr_n_split = splits(m)(s).n_elem;
        
        arma::field<arma::vec> tempsplits = splits(m);
        try{
          if(splits(m)(s).n_elem < 10){
            tempsplits = proposal_add_split(splits(m), s, y, X, p, n, base_model, lambda);
          }
          if(splits(m)(s).n_elem > curr_n_split){
            tot_added_splits++;
          }
          splits(m) = tempsplits;
        } catch(...){
          clog << "Warning, got error in MCMC [adding] -- staying at old splits." << endl;
          splits(m) = tempsplits;
        }
      }
    }
    if(move_type == 2){
      cout << "DROP SPLIT" << endl;
      for(int s=n_stages-1; s>-1; s--){
        cout << "  STAGE : " << s << endl;
        if(splits(m)(s).n_elem > 1){
          // can only drop on lower level with more than one element, or last level 
          unsigned int curr_n_split = splits(m)(s).n_elem;
          arma::field<arma::vec> tempsplits = splits(m);
          try{
            tempsplits = proposal_drop_split_2(splits(m), s, y, X, p, n, base_model, lambda);
            if(splits(m)(s).n_elem != curr_n_split){
              //clog << m << " " << s << " was different from before: " << splits(m)(s).n_elem << " vs " << curr_n_split << endl;
              tot_dropped_splits++;
            }
            splits(m) = tempsplits;
          } catch(...){
            clog << "Warning, got error in MCMC [drop] -- staying at old splits." << endl;
            splits(m) = tempsplits;
          }
        }
      }
      //cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }
    
    if(!onesigma){
      //if(move_type == 3){
        cout << "REFRESH" << endl;
        base_model = ModularLinReg(y, X, g, splits(m), radius, max_stages, -1.0, false, ain, bin, structpar);
    //}
    } else {
      Module lastmodule = base_model.modules[base_model.n_stages-1];
      arma::mat Px =  lastmodule.Xj * lastmodule.XtXi * lastmodule.Xj.t();
      double beta_post = arma::conv_to<double>::from(.5*(lastmodule.ej.t() * (lastmodule.In - lastmodule.g/(lastmodule.g+1.0) * lastmodule.Px) * lastmodule.ej));
      
      sigmasq_sample = 1.0 / bmrandom::rndpp_gamma(ain + n/2.0, 1.0/(bin + beta_post));
      //clog << sigmasq_sample << endl;
      base_model = ModularLinReg(y, X, g, splits(m), radius, max_stages, sigmasq_sample, false, ain, bin, structpar);
    }
    
    // burnin: find smoothing radius
    if(trysmooth){
      try{
        double radius_propose = exp( log(radius) + arma::randn()*.5 );
        if(abs(radius_propose) > 10){
          radius_propose = 10 * radius_propose/abs(radius_propose);
        }
        if(radius_propose != radius){
          ModularLinReg radius_update(y, X, g, splits(m), radius_propose, max_stages, -1.0, false, ain, bin, structpar);
          
          double prob = exp(arma::accu(radius_update.loglik - base_model.loglik)) * 
            exp(-radius_propose + radius) * radius_propose / radius;
          
          prob = prob > 1 ? 1 : prob;
          int accepted_proposal = bmrandom::rndpp_discrete({1-prob, prob});
          if(accepted_proposal == 1){
            radius = radius_propose;
            base_model = radius_update;
          } 
        } 
      } catch(...){
        clog << "Warning, got error in MCMC [radius] -- staying at old radius." << endl;
      }
    }
        /*
    if(move_type== 3){
      cout << "ADD STAGE" << endl;
      int old_stages = base_model.n_stages;
      
      splits(m) = proposal_add_stage(splits(m), y, X, p, n, base_model, max_stages);
      if(old_stages != base_model.n_stages){
        tot_added_stages++;
      }
      cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }
    if(move_type== 4){
      cout << "DROP STAGE" << endl;
      int old_stages = base_model.n_stages;
      if(n_stages>1){
        splits(m) = proposal_drop_stage(splits(m), base_model);
      } 
      if(old_stages != base_model.n_stages){
        tot_dropped_stages++;
      }
      cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }*/

    if(m > burn-1){
      int i = m-burn;
      theta_mcmc.row(i) = base_model.the_sample_field(base_model.n_stages-1).t();
      theta_ms(i) = base_model.the_sample_field;
      icept_mcmc(i) = arma::mean(y - X*theta_mcmc.row(i).t());
      radius_mcmc(i) = radius;
      //theta_mcmc.row(i) = base_model.the_sample_field(base_model.n_stages-1).t();
      //mu_mcmc.row(i) = base_model.mu_field(base_model.n_stages-1).t();
      //theta_ms(i) = base_model.the_sample_field;
      //sigmasq_ms.row(i) = base_model.sigmasq_scales.t();
      if(onesigma){
        sigmasq_mcmc(i) = sigmasq_sample;
      } else {
        sigmasq_mcmc(i) = base_model.sigmasq_scales(base_model.n_stages-1);
      }
      splits_save(i) = splits(m);
    }
    
    if(mcmc > 200){
      if(!(m % (mcmc/5))){
        clog << "[" << round(100*m/mcmc) << "%]";
      } 
    } 
    
  } // mcmc loop 
  clog << "[100%]" << endl;
  return Rcpp::List::create(
    Rcpp::Named("splits") = splits_save,
    //Rcpp::Named("mu") = mu_mcmc,
    //Rcpp::Named("mu_ms") = mu_ms,
    Rcpp::Named("intercept") = icept_mcmc,
    Rcpp::Named("radius") = radius_mcmc,
    Rcpp::Named("theta") = theta_mcmc,
    Rcpp::Named("theta_ms") = theta_ms,
    Rcpp::Named("sigmasq") = sigmasq_mcmc
    //Rcpp::Named("sigmasq_ms") = sigmasq_ms
  );
}


//' @export
// [[Rcpp::export]]
Rcpp::List sofk_binary(const arma::vec& y, const arma::mat& X, 
                       arma::field<arma::vec> start_splits, 
                       unsigned int mcmc = 100, unsigned int burn = 50,
                       double lambda=5.0,
                       int ii=0, int ll=0,
                       bool silent = true,
                       double structpar=10){
  if(silent){ cout.setstate(std::ios_base::failbit); } else {  cout.clear(); }
  // sample from posterior of changepoints given their number
  // idea:
  // each stage will use increasing number of changepoints
  // posterior from first used as offset
  
  int n = y.n_elem;
  int p = X.n_cols;
  int max_stages = start_splits.n_elem;
  
  double g=n/2.0;
  
  MCMCSWITCH = 1;
  
  arma::mat theta_mcmc = arma::zeros(mcmc - burn, p);
  arma::mat mu_mcmc = arma::zeros(mcmc - burn, p);
  arma::field<arma::field<arma::vec>> theta_ms = arma::field<arma::field<arma::vec>>(mcmc - burn);
  arma::field<arma::field<arma::vec>> mu_ms = arma::field<arma::field<arma::vec>>(mcmc - burn);
  
  arma::field<arma::field<arma::vec>> splits = arma::field<arma::field<arma::vec>>(mcmc);
  arma::field<arma::field<arma::vec>> splits_save = arma::field<arma::field<arma::vec>>(mcmc - burn);
  
  arma::mat sigmasq_ms = arma::zeros(mcmc - burn, max_stages);
  arma::mat sigmasq_mcmc = arma::vec(mcmc - burn);
  //arma::mat pred_mcmc = arma::zeros(mcmc, n);
  
  // initial field of splits is a simple split at 2
  splits(0) = arma::field<arma::vec>(max_stages);
  splits(0) = start_splits;//sample_no_replace(p-2, numsplits);
  //clog << splits(0)(0) << endl;
  //splits(0)(1) = arma::ones(1)*(p-3); 
  
  arma::field<arma::vec> current_splits = arma::field<arma::vec>(max_stages);
  //arma::field<arma::vec> proposed_splits = arma::field<arma::vec>(max_stages);
  
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
  
  
  int more_or_less = 0; // 0=move, 1=add/del
  int which_mol = 0; // index used to remove or add split at a stage
  
  int move_instage = 0; // -1=move to left, 1=move to right
  int move_outstage = 0; // 0=stay, 1=add a split to next stage
  
  double prob=0;
  int accepted_proposal;
  
  double proposed = 0;
  double accepted = 0;
  
  cout << "first model" << endl;
  ModularLinReg base_model(y, X, g, splits(0), 0, max_stages, 1.0, false, 2.1, 1.1, structpar);
  int m=0;
  
  ybin = y;
  z = X * base_model.the_sample_field(base_model.n_stages-1);
  
  cout << "first model done " << endl;
  int tot_added_splits = 0;
  int tot_dropped_splits = 0;
  int tot_moved = 0;
  int tot_added_stages = 0;
  int tot_dropped_stages = 0;
  double decay = 2.0;
  int choices = 4;
  
  for(unsigned int m = 1; m < mcmc; m++){
    Rcpp::checkUserInterrupt();
    cout << "========================================" << endl;
    cout << "starting from" << endl;
    cout << splits(m-1) << endl;
    cout << "----------------------------------------" << endl;
    cout << "MCMC : " << m << endl;
    // start by copying last splits. we will change it if proposals are accepted
    splits(m) = splits(m-1);
    
    int n_stages = base_model.n_stages;
    
    // move, add split, drop split, add stage, drop stage
    // this function only allows move
    
    int move_type = arma::randi<int>(arma::distr_param(0, choices-1));

    if(move_type == 0){    // cycle through the stages. for each stage we go through the splits
      cout << "MOVING [" << m << "]" << endl;
      for(unsigned int s=0; s<n_stages; s++){
        cout << "  STAGE : " << s << endl;
        
        // cycle through the splits. either change +1 or -1, or delete
        // for every proposal, choose whether to accept it
        // if accepted, save to splits at this mcmc stage
        for(unsigned int j=0; j<splits(m)(s).n_elem; j++){
          cout << "    SPLIT : " << j << endl;
          int curr_split = splits(m)(s)(j);
          splits(m) = proposal_move_split(splits(m), s, j, y, X, p, n, base_model, decay);
          proposed++;
          if(curr_split != splits(m)(s)(j)){
            tot_moved ++;
            accepted++;
          }
        } // split loop
      }
      cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }
    if(move_type == 1){
      cout << "ADD SPLIT" << endl;
      for(unsigned int s=0; s<n_stages; s++){
        cout << "  STAGE : " << s << endl;
        unsigned int curr_n_split = splits(m)(s).n_elem;
        splits(m) = proposal_add_split(splits(m), s, y, X, p, n, base_model, lambda);
        if(splits(m)(s).n_elem > curr_n_split){
          tot_added_splits++;
        }
      }
      cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }
    if(move_type == 2){
      cout << "DROP SPLIT" << endl;
      for(unsigned int s=0; s<n_stages; s++){
        cout << "  STAGE : " << s << endl;
        if((splits(m)(s).n_elem > 1) || ((splits(m)(s).n_elem == 1) & (s==n_stages-1) & s>0)){
          // can only drop on lower level with more than one element, or last level 
          unsigned int curr_n_split = splits(m)(s).n_elem;
          splits(m) = proposal_drop_split_2(splits(m), s, y, X, p, n, base_model, lambda);
          if(splits(m)(s).n_elem != curr_n_split){
            //clog << m << " " << s << " was different from before: " << splits(m)(s).n_elem << " vs " << curr_n_split << endl;
            tot_dropped_splits++;
          }
        }
      }
      cout << base_model.theta_p_scales(base_model.n_stages-1).t() << endl;
    }
    if(move_type == 3){
      cout << "REFRESH PARAMS" << endl;
      z = bmrandom::mvtruncnormal(base_model.intercept + X * base_model.the_sample_field(base_model.n_stages-1), trunc_lowerlim, trunc_upperlim, 1.0*In, 1).col(0);
      base_model = ModularLinReg(z, X, g, splits(m), 0, max_stages, 1.0, false,
                                 base_model.a, base_model.b, base_model.structpar);
    }
    
    if(m > burn-1){
      int i = m-burn;
      //clog << base_model.theta_p_scales << endl;
      
      theta_mcmc.row(i) = base_model.theta_p_scales(base_model.n_stages-1).t();
      theta_ms(i) = base_model.theta_p_scales;
      sigmasq_ms.row(i) = base_model.sigmasq_scales.t();
      sigmasq_mcmc(i) = base_model.sigmasq_scales(base_model.n_stages-1);
      splits_save(i) = splits(m);
    }
    
    if(mcmc > 100){
      if(!(m % 100)){
        decay = accepted/(proposed+0.0) < 0.3 ? decay+0.1 : decay-0.1;
        decay = decay<1 ? 1.0 : decay;
      } 
    }
    
    if(mcmc > 999){
      if(!(m % (mcmc/5))){
        cout << "[" << round(100*m/mcmc) << "%]";
      } 
    } 
    
  } // mcmc loop 
  
  clog << "[100%]" << endl;
  
  //clog << R::qnorm(0.025, -0.16, 1.0, 1, 0) << endl;
  return Rcpp::List::create(
    Rcpp::Named("splits") = splits_save,
    //Rcpp::Named("mu") = mu_mcmc,
    //Rcpp::Named("mu_ms") = mu_ms,
    Rcpp::Named("theta") = theta_mcmc,
    Rcpp::Named("theta_ms") = theta_ms,
    Rcpp::Named("sigmasq") = sigmasq_mcmc,
    Rcpp::Named("sigmasq_ms") = sigmasq_ms
  );
}

//' @export
// [[Rcpp::export]]
Rcpp::List bmms_base(arma::vec& y, arma::mat& X, 
                     double sigmasq,
                     double g, int mcmc, int burn, 
                     arma::field<arma::vec> splits, bool silent = true){
  // fixed splits, g prior
  if(silent){ cout.setstate(std::ios_base::failbit); } else {  cout.clear(); }
  
  int max_stages = 0;
  for(int i=0; i<splits.n_elem; i++){
    max_stages += splits(i).n_elem>0 ? 1 : 0;
  }
  
  int n = y.n_elem;
  int p = X.n_cols;
  
  arma::mat theta_mcmc = arma::zeros(mcmc-burn, p);
  arma::mat mu_mcmc = arma::zeros(mcmc-burn, p);
  arma::field<arma::field<arma::vec>> theta_ms = arma::field<arma::field<arma::vec>>(mcmc-burn);
  arma::field<arma::field<arma::vec>> mu_ms = arma::field<arma::field<arma::vec>>(mcmc-burn);
  
  arma::field<arma::vec> means_mcmc = arma::field<arma::vec>(mcmc-burn);
  //arma::cube theta_cov_mcmc = arma::zeros(p, p, mcmc-burn);
  arma::mat sigmasq_mcmc = arma::zeros(mcmc-burn, max_stages);
  arma::mat lmlik_mcmc = arma::zeros(mcmc-burn, max_stages);
  ModularLinReg base_model(y, X, g, splits, 0, max_stages, sigmasq, false,
                           2.1, 1.1, 0.0);
  
  int m=0;
  int final_stage = 1;
  
  //clog << "base done " << endl;
  
  for(int i=0; i<mcmc; i++){
    //if(!(i % (mcmc/100))){
    //  clog << floor(i*1.0/mcmc*100.0) << endl;
    //}
    //clog << i << endl;
    Rcpp::checkUserInterrupt();
    base_model.redo();
    
    if(i > burn-1){
      
      m = i-burn;
      //clog << "saving " << m << endl;
      final_stage = base_model.n_stages-1;
      theta_mcmc.row(m) = base_model.the_sample_field(final_stage).t();
      theta_ms(m) = base_model.the_sample_field;
      lmlik_mcmc.row(m) = base_model.loglik.t();
      //mu_ms(m) = base_model.mu_field;
      sigmasq_mcmc.row(m) = base_model.sigmasq_scales.t();
    }
    
  }
  
  //clog << "done " << endl;
  
  return Rcpp::List::create(
    Rcpp::Named("splits") = splits,
    //Rcpp::Named("mu_ms") = mu_ms,
    Rcpp::Named("theta") = theta_mcmc,
    Rcpp::Named("theta_ms") = theta_ms,
    Rcpp::Named("sigmasq") = sigmasq_mcmc,
    Rcpp::Named("loglik") = lmlik_mcmc );
}