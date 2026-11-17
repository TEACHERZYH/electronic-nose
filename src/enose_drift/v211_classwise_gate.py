"""v2.1.1 classwise gating with duplicate receipt-field compatibility."""
from __future__ import annotations
from typing import Mapping,Sequence,Hashable
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .evaluation import PredictionBundle
from .masking import availability_from_record,generate_evaluation_masks
from .v17_nonlinear import nonlinear_context_representation
from .v20_reconstruction import reconstruct_missing_groups,select_multimask_extratrees,select_reconstruction_model


BASE_NAMES=("ridge_reconstruct_rbf","ridge_reconstruct_shrinkage_lda","multimask_global_extratrees")


def classwise_recall_weights(probability_stack,truth,classes,beta,smoothing=1.0):
    """Return model-by-class weights from validation recalls."""
    stack=np.asarray(probability_stack,float); y=np.asarray(truth).astype(str); labels=np.asarray(classes).astype(str)
    if stack.ndim!=3 or stack.shape[0]!=len(y) or stack.shape[2]!=len(labels): raise ValueError("Classwise gate inputs are not aligned.")
    recalls=np.zeros((stack.shape[1],len(labels)),float)
    for model_index in range(stack.shape[1]):
        pred=labels[stack[:,model_index,:].argmax(1)]
        for class_index,label in enumerate(labels):
            target=y==label; tp=np.sum(pred[target]==label); recalls[model_index,class_index]=(tp+float(smoothing))/(target.sum()+2*float(smoothing))
    logits=float(beta)*recalls; logits-=logits.max(axis=0,keepdims=True); weights=np.exp(logits); weights/=weights.sum(axis=0,keepdims=True)
    return weights,recalls


def apply_classwise_gate(probability_stack,weights):
    stack=np.asarray(probability_stack,float); w=np.asarray(weights,float)
    if stack.ndim!=3 or w.shape!=stack.shape[1:]: raise ValueError("Classwise weights do not align with probabilities.")
    out=np.sum(stack*w[None,:,:],axis=1); return out/out.sum(axis=1,keepdims=True)


def _records(names,ranking,seed,counts,strategies):
    result=[]
    for strategy in strategies: result.extend(generate_evaluation_masks(names,counts,[int(seed)],strategy=strategy,influence_ranking=tuple(ranking) if strategy=="influence" else None))
    return result


def run_seed(seed,standardized,labels,sample_ids,dates,split_indices,groups,ranking,config,run_id):
    train,validation,test=split_indices; reconstruction=[]; selection=[]
    for key in ("ridge_rbf","ridge_lda"):
        selected,receipt=select_reconstruction_model(key,standardized[train],labels[train],standardized[validation],labels[validation],groups,ranking,config,seed); reconstruction.append(selected); selection.append(receipt)
    extra,centers,extra_receipt,training_receipt=select_multimask_extratrees(standardized[train],labels[train],standardized[validation],labels[validation],groups,ranking,config,seed); selection.append(extra_receipt)
    classes=np.asarray(reconstruction[0].estimator.classes_).astype(str)
    if any(np.asarray(model.estimator.classes_).astype(str).tolist()!=classes.tolist() for model in reconstruction) or np.asarray(extra.estimator.classes_).astype(str).tolist()!=classes.tolist(): raise ValueError("Base model classes differ.")
    names=tuple(groups); validation_records=_records(names,ranking,seed,config["validation_counts"],config["strategies"]); per_record=[]
    for record in validation_records:
        availability=availability_from_record(names,record); base=[]
        for selected in reconstruction:
            X=reconstruct_missing_groups(standardized[train],standardized[validation],groups,availability,alpha=selected.alpha); base.append(np.asarray(selected.estimator.predict_proba(X),float))
        X_extra=nonlinear_context_representation(standardized[validation],groups,availability,centers); base.append(np.asarray(extra.estimator.predict_proba(X_extra),float)); per_record.append(np.stack(base,axis=1))
    concatenated=np.concatenate(per_record,axis=0); repeated_truth=np.tile(np.asarray(labels[validation]).astype(str),len(validation_records)); beta_rows=[]; candidates=[]
    for beta in config["classwise_beta_candidates"]:
        weights,recalls=classwise_recall_weights(concatenated,repeated_truth,classes,float(beta),float(config["recall_smoothing"])); scores=[]; complete=[]
        for stack,record in zip(per_record,validation_records):
            probability=apply_classwise_gate(stack,weights); prediction=classes[probability.argmax(1)]; score=float(f1_score(labels[validation],prediction,average="macro",zero_division=0)); scores.append(score)
            if int(record.unavailable_count)==0: complete.append(score)
        row={"seed":int(seed),"beta":float(beta),"validation_worst_macro_f1":float(min(scores)),"validation_mean_macro_f1":float(np.mean(scores)),"validation_complete_macro_f1":float(np.mean(complete))}; beta_rows.append(row); candidates.append(((row["validation_worst_macro_f1"],row["validation_mean_macro_f1"],row["validation_complete_macro_f1"],-float(beta)),weights,recalls,row))
    best=max(candidates,key=lambda item:item[0]); weights,recalls,best_row=best[1],best[2],best[3]
    for row in beta_rows: row["selected"]=bool(row["beta"]==best_row["beta"])
    weight_rows=[]
    for model_index,method in enumerate(BASE_NAMES):
        for class_index,label in enumerate(classes): weight_rows.append({"seed":int(seed),"beta":float(best_row["beta"]),"method":method,"class":label,"validation_recall":float(recalls[model_index,class_index]),"classwise_weight":float(weights[model_index,class_index])})
    bundles=[]
    for strategy in config["strategies"]:
        records=generate_evaluation_masks(names,config["evaluation_counts"],[int(seed)],strategy=strategy,influence_ranking=tuple(ranking) if strategy=="influence" else None)
        for record in records:
            availability=availability_from_record(names,record); base=[]
            for selected in reconstruction:
                X=reconstruct_missing_groups(standardized[train],standardized[test],groups,availability,alpha=selected.alpha); base.append(np.asarray(selected.estimator.predict_proba(X),float))
            X_extra=nonlinear_context_representation(standardized[test],groups,availability,centers); base.append(np.asarray(extra.estimator.predict_proba(X_extra),float)); probability=apply_classwise_gate(np.stack(base,axis=1),weights); metadata={"mask_id":record.mask_id,"mask_strategy":strategy,"unavailable_count":int(record.unavailable_count),"selected_k":len(names),"selected_beta":float(best_row["beta"]),"test_labels_used_for_gate":False}
            bundle=PredictionBundle(run_id,"primary","validation_classwise_reconstruction_gate","test",int(seed),np.asarray(sample_ids[test]),np.asarray(dates[test]),np.asarray(labels[test]).astype(str),classes,probability,np.log(np.clip(probability,1e-15,1)),metadata); bundle.validate(); bundles.append(bundle)
    if "seed" not in training_receipt.columns:
        training_receipt.insert(0,"seed",int(seed))
    return bundles,pd.concat(selection,ignore_index=True),training_receipt,pd.DataFrame(beta_rows),pd.DataFrame(weight_rows)

