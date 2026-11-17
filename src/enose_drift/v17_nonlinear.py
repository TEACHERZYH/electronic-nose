"""Nonlinear multi-mask models for the frozen v1.7 exploratory study."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .evaluation import PredictionBundle
from .masking import (
    apply_group_availability,
    availability_from_record,
    generate_evaluation_masks,
    group_centers,
)
from .preprocessing import RobustTrainTransformer, select_group_columns
from .statistics import paired_date_cluster_bootstrap
from .v16_improvements import balanced_training_availability


METHOD_NAMES = {
    "rbf_svm": "multimask_rbf_svm",
    "hist_gbdt": "multimask_hist_gbdt",
    "mlp": "multimask_mlp_24",
}


def nonlinear_context_representation(
    X: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    availability: np.ndarray,
    centers: Mapping[Hashable, np.ndarray],
) -> np.ndarray:
    """Append sensor availability and its global fraction to center-filled features."""

    imputed = apply_group_availability(X, groups, availability, centers, append_mask=False)
    available = np.asarray(availability, dtype=bool)
    if available.ndim == 1:
        available = np.repeat(available[None, :], len(imputed), axis=0)
    if available.shape != (len(imputed), len(groups)):
        raise ValueError("Availability does not align with rows and sensor groups.")
    return np.hstack(
        [imputed, available.astype(float), available.mean(axis=1, keepdims=True)]
    )


def expand_nonlinear_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    *,
    random_counts: Sequence[int],
    seed: int,
    influence_ranking: Sequence[Hashable],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[Hashable, np.ndarray]]:
    availability, receipt = balanced_training_availability(
        len(X_train),
        tuple(groups),
        random_counts=random_counts,
        seed=int(seed),
        influence_ranking=influence_ranking,
    )
    source_rows = receipt["source_row"].to_numpy(dtype=int)
    centers = group_centers(X_train, groups)
    features = nonlinear_context_representation(
        np.asarray(X_train, dtype=float)[source_rows], groups, availability, centers
    )
    return features, np.asarray(y_train)[source_rows], receipt, centers


def _candidate_grid(model_key: str, model_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if model_key == "hist_gbdt":
        common = {
            "max_iter": int(model_config["max_iter"]),
            "learning_rate": float(model_config["learning_rate"]),
        }
        return [{**common, **dict(candidate)} for candidate in model_config["candidates"]]
    if model_key == "mlp":
        common = {
            "hidden_units": int(model_config["hidden_units"]),
            "max_iter": int(model_config["max_iter"]),
        }
        return [{**common, **dict(candidate)} for candidate in model_config["candidates"]]
    return [dict(candidate) for candidate in model_config["candidates"]]


def _make_estimator(model_key: str, parameters: Mapping[str, Any], *, seed: int):
    if model_key == "rbf_svm":
        return SVC(
            C=float(parameters["C"]),
            gamma=parameters["gamma"],
            kernel="rbf",
            probability=True,
            random_state=int(seed),
            cache_size=1024,
        )
    if model_key == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=int(parameters["max_iter"]),
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=int(seed),
        )
    if model_key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(int(parameters["hidden_units"]),),
            activation="relu",
            solver="adam",
            alpha=float(parameters["alpha"]),
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=int(parameters["max_iter"]),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=int(seed),
        )
    raise ValueError(f"Unknown nonlinear model key: {model_key}")


def _iteration_record(estimator, *, model_key: str, max_iter: int | None) -> dict[str, Any]:
    raw_n_iter = getattr(estimator, "n_iter_", None)
    if raw_n_iter is None:
        n_iter = None
    else:
        values = np.asarray(raw_n_iter).reshape(-1)
        n_iter = int(values.max()) if values.size else None
    fit_status = int(getattr(estimator, "fit_status_", 0))
    reached_max_iter = bool(
        model_key == "mlp" and max_iter is not None and n_iter is not None and n_iter >= max_iter
    )
    return {
        "n_iter": n_iter,
        "fit_status": fit_status,
        "reached_max_iter": reached_max_iter,
    }


@dataclass
class SelectedModel:
    model_key: str
    estimator: Any
    parameters: dict[str, Any]
    validation_macro_f1: float
    candidate_index: int


def tune_nonlinear_model(
    model_key: str,
    model_config: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int,
) -> tuple[SelectedModel | None, pd.DataFrame]:
    """Choose one eligible candidate using validation dates only."""

    records: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, int], SelectedModel]] = []
    for candidate_index, parameters in enumerate(_candidate_grid(model_key, model_config)):
        estimator = _make_estimator(model_key, parameters, seed=int(seed))
        caught: list[warnings.WarningMessage]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)
        probability = np.asarray(estimator.predict_proba(X_validation), dtype=float)
        finite_probability = bool(
            np.isfinite(probability).all()
            and np.all(probability >= 0.0)
            and np.allclose(probability.sum(axis=1), 1.0, atol=1e-7)
        )
        prediction = estimator.classes_[probability.argmax(axis=1)] if finite_probability else None
        score = (
            float(f1_score(y_validation, prediction, average="macro", zero_division=0))
            if prediction is not None
            else float("nan")
        )
        convergence_messages = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        iteration = _iteration_record(
            estimator,
            model_key=model_key,
            max_iter=int(parameters["max_iter"]) if "max_iter" in parameters else None,
        )
        eligible_fit = bool(
            finite_probability
            and np.isfinite(score)
            and iteration["fit_status"] == 0
            and not iteration["reached_max_iter"]
            and not convergence_messages
        )
        record = {
            "seed": int(seed),
            "model_key": model_key,
            "method": METHOD_NAMES[model_key],
            "candidate_index": int(candidate_index),
            "parameters": str(dict(parameters)),
            "validation_macro_f1": score,
            "finite_probability": finite_probability,
            "convergence_warning_count": len(convergence_messages),
            "convergence_messages": " | ".join(convergence_messages),
            "eligible": eligible_fit,
            **iteration,
        }
        records.append(record)
        if eligible_fit:
            selected = SelectedModel(
                model_key=model_key,
                estimator=estimator,
                parameters=dict(parameters),
                validation_macro_f1=score,
                candidate_index=int(candidate_index),
            )
            eligible.append(((score, -candidate_index), selected))
    if not eligible:
        return None, pd.DataFrame(records)
    best = max(eligible, key=lambda item: item[0])[1]
    for record in records:
        record["selected"] = bool(
            record["eligible"] and record["candidate_index"] == best.candidate_index
        )
    return best, pd.DataFrame(records)


def _bundle_from_estimator(
    selected: SelectedModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    run_id: str,
    seed: int,
    sample_ids: np.ndarray,
    date_ids: np.ndarray,
    metadata: dict[str, Any],
) -> PredictionBundle:
    probability = np.asarray(selected.estimator.predict_proba(X), dtype=float)
    bundle = PredictionBundle(
        run_id=run_id,
        dataset="primary",
        method=METHOD_NAMES[selected.model_key],
        split="test",
        seed=int(seed),
        sample_ids=np.asarray(sample_ids),
        date_ids=np.asarray(date_ids),
        y_true=np.asarray(y),
        classes=np.asarray(selected.estimator.classes_),
        probabilities=probability,
        logits=np.log(np.clip(probability, 1e-15, 1.0)),
        metadata=metadata,
    )
    bundle.validate()
    return bundle


def _entropy_gate(
    base: Sequence[PredictionBundle], *, reliability_floor: float, run_id: str
) -> PredictionBundle:
    if len(base) < 2:
        raise ValueError("Entropy gating requires at least two eligible base models.")
    first = base[0]
    for bundle in base[1:]:
        if (
            bundle.classes.tolist() != first.classes.tolist()
            or bundle.sample_ids.tolist() != first.sample_ids.tolist()
            or bundle.date_ids.tolist() != first.date_ids.tolist()
            or bundle.y_true.tolist() != first.y_true.tolist()
        ):
            raise ValueError("Entropy-gated base predictions are not sample-aligned.")
    stack = np.stack([bundle.probabilities for bundle in base], axis=1)
    entropy = -np.sum(stack * np.log(np.clip(stack, 1e-15, 1.0)), axis=2)
    reliability = np.maximum(float(reliability_floor), 1.0 - entropy / np.log(stack.shape[2]))
    weights = reliability / reliability.sum(axis=1, keepdims=True)
    probability = np.sum(stack * weights[:, :, None], axis=1)
    metadata = dict(first.metadata or {})
    metadata.update(
        {
            "base_methods": [bundle.method for bundle in base],
            "reliability_floor": float(reliability_floor),
            "test_labels_used_for_gate": False,
        }
    )
    result = PredictionBundle(
        run_id=run_id,
        dataset="primary",
        method="entropy_gated_nonlinear_ensemble",
        split="test",
        seed=first.seed,
        sample_ids=first.sample_ids,
        date_ids=first.date_ids,
        y_true=first.y_true,
        classes=first.classes,
        probabilities=probability,
        logits=np.log(np.clip(probability, 1e-15, 1.0)),
        metadata=metadata,
    )
    result.validate()
    return result


def run_seed(
    seed: int,
    standardized: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    dates: np.ndarray,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    compact_groups: Mapping[Hashable, Sequence[int]],
    influence_ranking: Sequence[Hashable],
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[list[PredictionBundle], pd.DataFrame, pd.DataFrame]:
    train, validation, test = split_indices
    expanded_X, expanded_y, training_receipt, centers = expand_nonlinear_training(
        standardized[train],
        labels[train],
        compact_groups,
        random_counts=config["training_masks"]["random_counts"],
        seed=int(seed),
        influence_ranking=influence_ranking,
    )
    training_receipt.insert(0, "seed", int(seed))
    complete = np.ones((len(validation), len(compact_groups)), dtype=bool)
    validation_X = nonlinear_context_representation(
        standardized[validation], compact_groups, complete, centers
    )
    selected_models: list[SelectedModel] = []
    convergence: list[pd.DataFrame] = []
    for model_key in ("rbf_svm", "hist_gbdt", "mlp"):
        selected, receipt = tune_nonlinear_model(
            model_key,
            config["models"][model_key],
            expanded_X,
            expanded_y,
            validation_X,
            labels[validation],
            seed=int(seed),
        )
        convergence.append(receipt)
        if selected is not None:
            selected_models.append(selected)

    bundles: list[PredictionBundle] = []
    selected_names = tuple(influence_ranking)
    for strategy in config["evaluation_masks"]["strategies"]:
        records = generate_evaluation_masks(
            selected_names,
            config["evaluation_masks"]["unavailable_counts"],
            [int(seed)],
            strategy=str(strategy),
            influence_ranking=selected_names if str(strategy) == "influence" else None,
        )
        for record in records:
            availability = availability_from_record(selected_names, record)
            test_X = nonlinear_context_representation(
                standardized[test], compact_groups, availability, centers
            )
            one_mask: list[PredictionBundle] = []
            for selected in selected_models:
                metadata = {
                    "mask_id": record.mask_id,
                    "mask_strategy": str(strategy),
                    "unavailable_count": int(record.unavailable_count),
                    "selected_k": len(selected_names),
                    "candidate_index": selected.candidate_index,
                    "selected_parameters": selected.parameters,
                    "validation_macro_f1": selected.validation_macro_f1,
                    "training_patterns_per_row": int(len(training_receipt) / len(train)),
                    "convergence_eligible": True,
                }
                bundle = _bundle_from_estimator(
                    selected,
                    test_X,
                    labels[test],
                    run_id=run_id,
                    seed=int(seed),
                    sample_ids=sample_ids[test],
                    date_ids=dates[test],
                    metadata=metadata,
                )
                bundles.append(bundle)
                one_mask.append(bundle)
            if len(one_mask) >= 2:
                bundles.append(
                    _entropy_gate(
                        one_mask,
                        reliability_floor=float(
                            config["models"]["entropy_gate"]["reliability_floor"]
                        ),
                        run_id=run_id,
                    )
                )
    return bundles, training_receipt, pd.concat(convergence, ignore_index=True)


def prepare_v17_inputs(
    X: np.ndarray,
    manifest: pd.DataFrame,
    groups: Mapping[Hashable, Sequence[int]],
    influence_ranking: Sequence[Hashable],
    *,
    always_include_columns: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    required = {"split", "label", "sample_id", "date"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"v1.7 manifest lacks {sorted(required - set(manifest.columns))}.")
    selected = tuple(influence_ranking)
    columns, compact_groups = select_group_columns(dict(groups), selected)
    always = np.asarray(tuple(int(value) for value in always_include_columns), dtype=int)
    if always.size:
        columns = np.concatenate([columns, always])
    selected_X = np.asarray(X[:, columns], dtype=float)
    train = np.flatnonzero(manifest["split"].eq("train").to_numpy())
    validation = np.flatnonzero(manifest["split"].eq("validation").to_numpy())
    test = np.flatnonzero(manifest["split"].eq("test").to_numpy())
    transformer = RobustTrainTransformer().fit(selected_X[train])
    standardized = transformer.transform(selected_X)
    return (
        standardized,
        manifest["label"].astype(str).to_numpy(),
        manifest["sample_id"].to_numpy(),
        manifest["date"].to_numpy(),
        (train, validation, test),
        compact_groups,
    )


def validate_baseline_alignment(new_predictions: pd.DataFrame, baseline: pd.DataFrame) -> None:
    keys = ["seed", "sample_id", "date_id", "y_true", "mask_id", "mask_strategy", "unavailable_count"]
    new_units = new_predictions.loc[:, keys].drop_duplicates()
    baseline_units = baseline.loc[:, keys].drop_duplicates()
    merged = new_units.merge(baseline_units, on=keys, how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("New and frozen baseline predictions are not exactly mask/sample aligned.")


def summarize_comparisons(
    date_metrics: pd.DataFrame,
    baseline_date_metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    combined = pd.concat([date_metrics, baseline_date_metrics], ignore_index=True, sort=False)
    metrics = ["macro_f1", "worst_class_recall"]
    averaged = (
        combined.groupby(
            ["method", "mask_strategy", "unavailable_count", "date_id"],
            as_index=False,
            sort=False,
        )[metrics]
        .mean()
    )
    methods = sorted(set(date_metrics["method"]))
    comparisons: dict[str, Any] = {}
    for method in methods:
        comparisons[method] = {}
        for strategy in ("random", "influence"):
            comparisons[method][strategy] = {}
            for count in (0, 16):
                left = averaged.loc[
                    averaged["method"].eq(method)
                    & averaged["mask_strategy"].eq(strategy)
                    & averaged["unavailable_count"].eq(count)
                ].set_index("date_id")
                for reference in ("unconditioned", "TCSS-mask-aware"):
                    right = averaged.loc[
                        averaged["method"].eq(reference)
                        & averaged["mask_strategy"].eq(strategy)
                        & averaged["unavailable_count"].eq(count)
                    ].set_index("date_id")
                    if set(left.index) != set(right.index):
                        raise ValueError("Date identities differ in a paired v1.7 comparison.")
                    units = sorted(left.index.astype(str))
                    result = paired_date_cluster_bootstrap(
                        [float(left.loc[unit, "macro_f1"]) for unit in units],
                        [float(right.loc[unit, "macro_f1"]) for unit in units],
                        dates=units,
                        n_resamples=int(bootstrap_replicates),
                        confidence_level=float(confidence_level),
                        seed=int(bootstrap_seed),
                    )
                    result["mean_worst_class_recall"] = float(left["worst_class_recall"].mean())
                    result["reference_mean_worst_class_recall"] = float(
                        right["worst_class_recall"].mean()
                    )
                    comparisons[method][strategy][f"count{count}_vs_{reference}"] = result
    return {"comparisons": comparisons, "date_averaged_metrics": averaged.to_dict("records")}

