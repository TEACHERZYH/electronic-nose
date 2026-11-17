"""Cross-sensor reconstruction and global interaction models for v2.0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from .evaluation import PredictionBundle
from .masking import availability_from_record, generate_evaluation_masks
from .v17_nonlinear import expand_nonlinear_training, nonlinear_context_representation


METHOD_NAMES = {
    "ridge_rbf": "ridge_reconstruct_rbf",
    "ridge_lda": "ridge_reconstruct_shrinkage_lda",
    "multimask_extra": "multimask_global_extratrees",
}


def reconstruct_missing_groups(
    X_train: np.ndarray,
    X_target: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    availability: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Reconstruct unavailable group columns using training rows only."""
    train = np.asarray(X_train, dtype=float)
    target = np.asarray(X_target, dtype=float)
    available = np.asarray(availability, dtype=bool)
    if available.shape != (len(groups),):
        raise ValueError("Availability must contain one value per sensor group.")
    missing_parts = [np.asarray(indices, dtype=int) for flag, indices in zip(available, groups.values()) if not flag]
    if not missing_parts:
        return target.copy()
    missing = np.unique(np.concatenate(missing_parts))
    observed = np.setdiff1d(np.arange(train.shape[1], dtype=int), missing, assume_unique=True)
    if observed.size == 0:
        raise ValueError("Reconstruction requires at least one observed feature.")
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(train[:, observed], train[:, missing])
    result = target.copy()
    predicted = np.asarray(model.predict(target[:, observed]), dtype=float)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    result[:, missing] = predicted
    if not np.isfinite(result).all():
        raise ValueError("Reconstruction produced non-finite values.")
    return result


def _validation_records(names, ranking, seed, counts, strategies):
    records=[]
    for strategy in strategies:
        records.extend(generate_evaluation_masks(names, counts, [int(seed)], strategy=strategy, influence_ranking=tuple(ranking) if strategy=="influence" else None))
    return records


def _robust_scores(probabilities, truth, records):
    scores=[]; complete=[]
    for probability,record in zip(probabilities,records):
        prediction=np.asarray(probability[1])[np.asarray(probability[0]).argmax(1)]
        score=float(f1_score(truth,prediction,average="macro",zero_division=0)); scores.append(score)
        if int(record.unavailable_count)==0: complete.append(score)
    return float(min(scores)),float(np.mean(scores)),float(np.mean(complete))


def _make_complete_estimator(model_key,parameters,seed):
    if model_key=="ridge_rbf":
        return SVC(C=float(parameters["C"]),gamma="scale",kernel="rbf",probability=True,cache_size=1024,random_state=int(seed))
    if model_key=="ridge_lda":
        return LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
    raise ValueError(model_key)


@dataclass
class SelectedReconstructionModel:
    model_key: str
    estimator: Any
    alpha: float
    parameters: dict[str, Any]
    validation_worst_macro_f1: float
    validation_mean_macro_f1: float


def select_reconstruction_model(model_key,X_train,y_train,X_validation,y_validation,groups,ranking,config,seed):
    names=tuple(groups); records=_validation_records(names,ranking,seed,config["validation_counts"],config["strategies"])
    reconstructed={}
    for alpha in config["ridge_alphas"]:
        reconstructed[float(alpha)]=[
            reconstruct_missing_groups(X_train,X_validation,groups,availability_from_record(names,record),alpha=float(alpha)) for record in records
        ]
    candidates=config["models"][model_key]["candidates"]
    rows=[]; eligible=[]
    for candidate_index,parameters in enumerate(candidates):
        estimator=_make_complete_estimator(model_key,parameters,seed).fit(X_train,y_train)
        classes=np.asarray(estimator.classes_).astype(str)
        for alpha in config["ridge_alphas"]:
            probs=[(np.asarray(estimator.predict_proba(X),float),classes) for X in reconstructed[float(alpha)]]
            worst,mean,complete=_robust_scores(probs,np.asarray(y_validation).astype(str),records)
            row={"seed":int(seed),"model_key":model_key,"method":METHOD_NAMES[model_key],"candidate_index":int(candidate_index),"parameters":str(dict(parameters)),"alpha":float(alpha),"validation_worst_macro_f1":worst,"validation_mean_macro_f1":mean,"validation_complete_macro_f1":complete}
            rows.append(row); eligible.append(((worst,mean,complete,-candidate_index,-float(alpha)),estimator,row))
    best=max(eligible,key=lambda item:item[0]); selected=SelectedReconstructionModel(model_key,best[1],float(best[2]["alpha"]),dict(candidates[int(best[2]["candidate_index"])]),float(best[2]["validation_worst_macro_f1"]),float(best[2]["validation_mean_macro_f1"]))
    for row in rows: row["selected"]=bool(row["candidate_index"]==best[2]["candidate_index"] and row["alpha"]==best[2]["alpha"])
    return selected,pd.DataFrame(rows)


@dataclass
class SelectedExtraTrees:
    estimator: ExtraTreesClassifier
    parameters: dict[str, Any]
    validation_worst_macro_f1: float
    validation_mean_macro_f1: float


def select_multimask_extratrees(X_train,y_train,X_validation,y_validation,groups,ranking,config,seed):
    expanded_X,expanded_y,training_receipt,centers=expand_nonlinear_training(X_train,y_train,groups,random_counts=config["training_counts"],seed=int(seed),influence_ranking=ranking)
    names=tuple(groups); records=_validation_records(names,ranking,seed,config["validation_counts"],config["strategies"])
    validation_features=[nonlinear_context_representation(X_validation,groups,availability_from_record(names,record),centers) for record in records]
    rows=[]; eligible=[]
    for candidate_index,parameters in enumerate(config["models"]["multimask_extra"]["candidates"]):
        estimator=ExtraTreesClassifier(n_estimators=int(config["models"]["multimask_extra"]["n_estimators"]),max_features=parameters["max_features"],min_samples_leaf=int(parameters["min_samples_leaf"]),class_weight="balanced",n_jobs=1,random_state=int(seed)).fit(expanded_X,expanded_y)
        classes=np.asarray(estimator.classes_).astype(str); probs=[(np.asarray(estimator.predict_proba(X),float),classes) for X in validation_features]
        worst,mean,complete=_robust_scores(probs,np.asarray(y_validation).astype(str),records)
        row={"seed":int(seed),"model_key":"multimask_extra","method":METHOD_NAMES["multimask_extra"],"candidate_index":int(candidate_index),"parameters":str(dict(parameters)),"validation_worst_macro_f1":worst,"validation_mean_macro_f1":mean,"validation_complete_macro_f1":complete}
        rows.append(row); eligible.append(((worst,mean,complete,-candidate_index),estimator,row))
    best=max(eligible,key=lambda item:item[0]); selected=SelectedExtraTrees(best[1],dict(config["models"]["multimask_extra"]["candidates"][int(best[2]["candidate_index"])]),float(best[2]["validation_worst_macro_f1"]),float(best[2]["validation_mean_macro_f1"]))
    for row in rows: row["selected"]=bool(row["candidate_index"]==best[2]["candidate_index"])
    training_receipt.insert(0,"seed",int(seed))
    return selected,centers,pd.DataFrame(rows),training_receipt


def _bundle(method,estimator,X,y,sample_ids,dates,seed,run_id,metadata):
    probability=np.asarray(estimator.predict_proba(X),float); classes=np.asarray(estimator.classes_).astype(str)
    bundle=PredictionBundle(run_id,"primary",method,"test",int(seed),np.asarray(sample_ids),np.asarray(dates),np.asarray(y).astype(str),classes,probability,np.log(np.clip(probability,1e-15,1)),metadata)
    bundle.validate(); return bundle


def _entropy_gate(base,run_id,reliability_floor):
    first=base[0]
    for bundle in base[1:]:
        if bundle.classes.tolist()!=first.classes.tolist() or bundle.sample_ids.tolist()!=first.sample_ids.tolist() or bundle.y_true.tolist()!=first.y_true.tolist():
            raise ValueError("Entropy-gated reconstruction models are not aligned.")
    stack=np.stack([b.probabilities for b in base],axis=1)
    entropy=-np.sum(stack*np.log(np.clip(stack,1e-15,1)),axis=2); reliability=np.maximum(float(reliability_floor),1-entropy/np.log(stack.shape[2])); weights=reliability/reliability.sum(1,keepdims=True); probability=np.sum(stack*weights[:,:,None],axis=1)
    metadata=dict(first.metadata or {}); metadata.update({"base_methods":[b.method for b in base],"test_labels_used_for_gate":False})
    bundle=PredictionBundle(run_id,"primary","entropy_gated_reconstruction_ensemble","test",first.seed,first.sample_ids,first.date_ids,first.y_true,first.classes,probability,np.log(np.clip(probability,1e-15,1)),metadata); bundle.validate(); return bundle


def run_seed(seed,standardized,labels,sample_ids,dates,split_indices,groups,ranking,config,run_id):
    train,validation,test=split_indices
    reconstruction=[]; receipts=[]
    for model_key in ("ridge_rbf","ridge_lda"):
        selected,receipt=select_reconstruction_model(model_key,standardized[train],labels[train],standardized[validation],labels[validation],groups,ranking,config,seed); reconstruction.append(selected); receipts.append(receipt)
    extra,centers,extra_receipt,training_receipt=select_multimask_extratrees(standardized[train],labels[train],standardized[validation],labels[validation],groups,ranking,config,seed); receipts.append(extra_receipt)
    bundles=[]; names=tuple(groups)
    for strategy in config["strategies"]:
        records=generate_evaluation_masks(names,config["evaluation_counts"],[int(seed)],strategy=strategy,influence_ranking=tuple(ranking) if strategy=="influence" else None)
        for record in records:
            availability=availability_from_record(names,record); metadata={"mask_id":record.mask_id,"mask_strategy":strategy,"unavailable_count":int(record.unavailable_count),"selected_k":len(names)}; one=[]
            for selected in reconstruction:
                X=reconstruct_missing_groups(standardized[train],standardized[test],groups,availability,alpha=selected.alpha); meta={**metadata,"ridge_alpha":selected.alpha,"selected_parameters":selected.parameters,"validation_worst_macro_f1":selected.validation_worst_macro_f1,"validation_mean_macro_f1":selected.validation_mean_macro_f1}; one.append(_bundle(METHOD_NAMES[selected.model_key],selected.estimator,X,labels[test],sample_ids[test],dates[test],seed,run_id,meta))
            X_extra=nonlinear_context_representation(standardized[test],groups,availability,centers); meta={**metadata,"selected_parameters":extra.parameters,"validation_worst_macro_f1":extra.validation_worst_macro_f1,"validation_mean_macro_f1":extra.validation_mean_macro_f1}; one.append(_bundle(METHOD_NAMES["multimask_extra"],extra.estimator,X_extra,labels[test],sample_ids[test],dates[test],seed,run_id,meta))
            bundles.extend(one); bundles.append(_entropy_gate(one,run_id,config["entropy_reliability_floor"]))
    return bundles,pd.concat(receipts,ignore_index=True),training_receipt
