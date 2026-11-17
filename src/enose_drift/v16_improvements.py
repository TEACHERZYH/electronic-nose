"""Post-result exploratory improvements for the v1.6 existing-data study.

The routines consume the immutable ``formal_02`` prediction artifacts.  They
never relabel the source test sets as independent confirmation and never tune
on source test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .calibration import TemperatureScaler, softmax_temperature
from .evaluation import PredictionBundle, bundle_from_model
from .masking import (
    apply_group_availability,
    availability_from_record,
    generate_evaluation_masks,
    group_centers,
)
from .metrics import classification_metrics, probability_metrics
from .preprocessing import RobustTrainTransformer, select_group_columns
from .statistics import paired_date_cluster_bootstrap


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("p_")]
    if not columns:
        raise ValueError("Prediction frame has no probability columns.")
    probability = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("Prediction probabilities must be finite and nonnegative.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Prediction probability rows must sum to one.")
    return columns


def labels_from_probability_columns(columns: Sequence[str]) -> list[str]:
    return [str(column)[2:] for column in columns]


def _score_frame(frame: pd.DataFrame, *, supported_only: bool = False) -> float:
    columns = probability_columns(frame)
    labels = labels_from_probability_columns(columns)
    probability = frame.loc[:, columns].to_numpy(dtype=float)
    prediction = np.asarray(labels, dtype=object)[probability.argmax(axis=1)]
    return float(
        classification_metrics(
            frame["y_true"].astype(str).to_numpy(),
            prediction,
            labels=labels,
            supported_only=supported_only,
        )["macro_f1"]
    )


def metric_records(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    supported_only: bool = False,
    coverages: Sequence[float] = (0.25, 0.50, 0.75),
) -> pd.DataFrame:
    probability_cols = probability_columns(frame)
    labels = labels_from_probability_columns(probability_cols)
    records: list[dict] = []
    grouper = list(group_columns)
    key_arg = grouper[0] if len(grouper) == 1 else grouper
    for key, one in frame.groupby(key_arg, sort=False, dropna=False):
        key_values = (key,) if len(grouper) == 1 else tuple(key)
        probability = one.loc[:, probability_cols].to_numpy(dtype=float)
        truth = one["y_true"].astype(str).to_numpy()
        prediction = np.asarray(labels, dtype=object)[probability.argmax(axis=1)]
        classification = classification_metrics(
            truth,
            prediction,
            labels=labels,
            supported_only=supported_only,
        )
        probability_result = probability_metrics(truth, probability, labels=labels)
        record = {column: value for column, value in zip(grouper, key_values)}
        record.update(
            {
                "n": int(len(one)),
                "macro_f1": float(classification["macro_f1"]),
                "worst_class_recall": float(np.min(classification["class_recall"])),
                "nll": float(probability_result["nll"]),
                "brier": float(probability_result["brier"]),
                "ece": float(probability_result["ece"]),
                "aurc": float(probability_result["aurc"]),
            }
        )
        confidence = probability.max(axis=1)
        correct = prediction == truth
        order = np.argsort(-confidence, kind="stable")
        for coverage in coverages:
            keep = max(1, int(np.ceil(float(coverage) * len(one))))
            record[f"risk_at_{int(round(100 * float(coverage)))}pct"] = float(
                1.0 - np.mean(correct[order[:keep]])
            )
        records.append(record)
    return pd.DataFrame(records)


def seed_average(frame: pd.DataFrame, *, group_columns: Sequence[str]) -> pd.DataFrame:
    metrics = [
        column
        for column in frame.columns
        if column in {"macro_f1", "worst_class_recall", "nll", "brier", "ece", "aurc"}
        or column.startswith("risk_at_")
    ]
    return frame.groupby(list(group_columns), sort=False, as_index=False)[metrics].mean()


def paired_summary(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    unit_column: str,
    metric: str,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict:
    a = first.set_index(unit_column)[metric]
    b = second.set_index(unit_column)[metric]
    if set(a.index) != set(b.index):
        raise ValueError(f"Paired {unit_column} identities differ.")
    units = list(a.index)
    return paired_date_cluster_bootstrap(
        [float(a.loc[unit]) for unit in units],
        [float(b.loc[unit]) for unit in units],
        dates=units,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def run_i3_validation_gate(
    resource_predictions: pd.DataFrame,
    *,
    sensor_budget: int,
    blank_budget: int,
    margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = resource_predictions.copy()
    required = {"method", "split", "seed", "sample_id", "date_id", "y_true"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Resource predictions lack {sorted(required - set(frame.columns))}.")
    selected = frame.loc[
        pd.to_numeric(frame["sensor_budget"], errors="coerce").eq(sensor_budget)
        & pd.to_numeric(frame["blank_budget"], errors="coerce").eq(blank_budget)
    ].copy()
    decisions: list[dict] = []
    output: list[pd.DataFrame] = []
    for seed_value in sorted(pd.to_numeric(selected["seed"], errors="raise").unique()):
        seed = int(seed_value)
        validation_scores: dict[str, float] = {}
        for method in ("TCSS", "pooled_group_lasso"):
            validation = selected.loc[
                selected["seed"].eq(seed)
                & selected["split"].eq("validation")
                & selected["method"].eq(method)
            ]
            if validation.empty:
                raise ValueError(f"I3 lacks validation predictions for seed={seed}, method={method}.")
            validation_scores[method] = _score_frame(validation)
        choose_tcss = validation_scores["TCSS"] > validation_scores["pooled_group_lasso"] + margin
        chosen = "TCSS" if choose_tcss else "pooled_group_lasso"
        test = selected.loc[
            selected["seed"].eq(seed)
            & selected["split"].eq("test")
            & selected["method"].eq(chosen)
        ].copy()
        if test.empty:
            raise ValueError(f"I3 lacks test predictions for chosen method {chosen}.")
        test["source_method"] = chosen
        test["method"] = "validation_gated"
        output.append(test)
        decisions.append(
            {
                "seed": seed,
                "tcss_validation_macro_f1": validation_scores["TCSS"],
                "pooled_validation_macro_f1": validation_scores["pooled_group_lasso"],
                "margin": float(margin),
                "selected_method": chosen,
                "test_labels_used_for_decision": False,
            }
        )
    gated = pd.concat(output, ignore_index=True)
    pooled = selected.loc[selected["split"].eq("test") & selected["method"].eq("pooled_group_lasso")].copy()
    return gated, pooled, pd.DataFrame(decisions)


def run_i4_fixed_ensemble(
    mask_predictions: pd.DataFrame,
    *,
    mask_aware_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= mask_aware_weight <= 1.0:
        raise ValueError("mask_aware_weight must lie in [0, 1].")
    frame = mask_predictions.copy()
    probability_cols = probability_columns(frame)
    keys = [
        "seed",
        "sample_id",
        "date_id",
        "y_true",
        "mask_id",
        "mask_strategy",
        "unavailable_count",
        "selected_k",
    ]
    aware = frame.loc[frame["method"].eq("TCSS-mask-aware"), keys + probability_cols].copy()
    plain = frame.loc[frame["method"].eq("unconditioned"), keys + probability_cols].copy()
    merged = aware.merge(plain, on=keys, suffixes=("_aware", "_plain"), validate="one_to_one")
    if len(merged) != len(aware) or len(merged) != len(plain):
        raise ValueError("I4 methods do not share exactly the same masks and samples.")
    output = merged.loc[:, keys].copy()
    for column in probability_cols:
        output[column] = (
            mask_aware_weight * merged[f"{column}_aware"]
            + (1.0 - mask_aware_weight) * merged[f"{column}_plain"]
        )
    output["method"] = "fixed_0.5_mask_ensemble"
    output["mask_aware_weight"] = float(mask_aware_weight)
    output["y_pred"] = np.asarray(labels_from_probability_columns(probability_cols), dtype=object)[
        output.loc[:, probability_cols].to_numpy().argmax(axis=1)
    ]
    receipts = output.loc[
        :, ["seed", "mask_id", "mask_strategy", "unavailable_count", "mask_aware_weight"]
    ].drop_duplicates()
    return output, receipts.reset_index(drop=True)


@dataclass
class VectorScaler:
    l2: float = 1.0

    def fit(self, logits: np.ndarray, y: Sequence[str], classes: Sequence[str]) -> "VectorScaler":
        values = np.asarray(logits, dtype=float)
        truth = np.asarray(y).astype(str)
        self.classes_ = np.asarray(classes).astype(str)
        if values.shape != (len(truth), len(self.classes_)):
            raise ValueError("Vector scaling inputs are not aligned.")
        class_index = {label: index for index, label in enumerate(self.classes_)}
        y_index = np.asarray([class_index[label] for label in truth], dtype=int)
        n_classes = len(self.classes_)

        def objective(parameters: np.ndarray) -> float:
            log_scale = parameters[:n_classes]
            bias = parameters[n_classes:] - np.mean(parameters[n_classes:])
            adjusted = values * np.exp(log_scale)[None, :] + bias[None, :]
            adjusted -= adjusted.max(axis=1, keepdims=True)
            probability = np.exp(np.clip(adjusted, -745.0, 50.0))
            probability /= probability.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(np.clip(probability[np.arange(len(truth)), y_index], 1e-15, 1.0)))
            penalty = float(self.l2) * float(np.mean(log_scale**2) + np.mean(bias**2))
            return float(nll + penalty)

        result = minimize(objective, np.zeros(2 * n_classes), method="L-BFGS-B")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Vector scaling failed: {result.message}")
        self.log_scale_ = np.asarray(result.x[:n_classes], dtype=float)
        raw_bias = np.asarray(result.x[n_classes:], dtype=float)
        self.bias_ = raw_bias - raw_bias.mean()
        self.validation_objective_ = float(result.fun)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not hasattr(self, "log_scale_"):
            raise RuntimeError("VectorScaler must be fitted before prediction.")
        adjusted = np.asarray(logits, dtype=float) * np.exp(self.log_scale_)[None, :] + self.bias_[None, :]
        adjusted -= adjusted.max(axis=1, keepdims=True)
        probability = np.exp(np.clip(adjusted, -745.0, 50.0))
        return probability / probability.sum(axis=1, keepdims=True)


def select_vector_l2(
    validation: pd.DataFrame,
    *,
    candidates: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    probability_cols = probability_columns(validation)
    classes = labels_from_probability_columns(probability_cols)
    logits = np.log(np.clip(validation.loc[:, probability_cols].to_numpy(dtype=float), 1e-15, 1.0))
    dates = validation["date_id"].astype(str).to_numpy()
    truth = validation["y_true"].astype(str).to_numpy()
    records: list[dict] = []
    for l2 in candidates:
        fold_losses: list[float] = []
        for held_out in pd.unique(dates):
            training = dates != held_out
            testing = ~training
            scaler = VectorScaler(float(l2)).fit(logits[training], truth[training], classes)
            probability = scaler.predict_proba(logits[testing])
            loss = probability_metrics(truth[testing], probability, labels=classes)["nll"]
            fold_losses.append(float(loss))
            records.append({"l2": float(l2), "held_out_date": held_out, "nll": float(loss)})
        records.append({"l2": float(l2), "held_out_date": "MEAN", "nll": float(np.mean(fold_losses))})
    receipt = pd.DataFrame(records)
    means = receipt.loc[receipt["held_out_date"].eq("MEAN")]
    selected = means.sort_values(["nll", "l2"], ascending=[True, False], kind="stable").iloc[0]
    return float(selected["l2"]), receipt


def run_i5_calibration(
    resource_predictions: pd.DataFrame,
    *,
    sensor_budget: int,
    blank_budget: int,
    l2_candidates: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = resource_predictions.loc[
        pd.to_numeric(resource_predictions["sensor_budget"], errors="coerce").eq(sensor_budget)
        & pd.to_numeric(resource_predictions["blank_budget"], errors="coerce").eq(blank_budget)
    ].copy()
    probability_cols = probability_columns(frame)
    classes = labels_from_probability_columns(probability_cols)
    outputs: list[pd.DataFrame] = []
    receipts: list[pd.DataFrame] = []
    for seed_value in sorted(pd.to_numeric(frame["seed"], errors="raise").unique()):
        seed = int(seed_value)
        validation = frame.loc[
            frame["seed"].eq(seed) & frame["split"].eq("validation") & frame["method"].eq("TCSS")
        ].copy()
        target = frame.loc[
            frame["seed"].eq(seed) & frame["split"].eq("test") & frame["method"].eq("TCSS-unscaled")
        ].copy()
        if validation.empty or target.empty:
            raise ValueError(f"I5 lacks validation or target predictions for seed={seed}.")
        validation_logits = np.log(np.clip(validation.loc[:, probability_cols].to_numpy(float), 1e-15, 1.0))
        target_logits = np.log(np.clip(target.loc[:, probability_cols].to_numpy(float), 1e-15, 1.0))
        selected_l2, receipt = select_vector_l2(validation, candidates=l2_candidates)
        receipt.insert(0, "seed", seed)
        receipts.append(receipt)
        temperature = TemperatureScaler().fit(
            validation_logits,
            validation["y_true"].astype(str).to_numpy(),
            np.asarray(classes, dtype=object),
        )
        vector = VectorScaler(selected_l2).fit(
            validation_logits,
            validation["y_true"].astype(str).to_numpy(),
            classes,
        )
        variants = {
            "unscaled": target.loc[:, probability_cols].to_numpy(float),
            "temperature": softmax_temperature(target_logits, temperature.temperature_),
            "vector": vector.predict_proba(target_logits),
        }
        for method, probability in variants.items():
            one = target.loc[:, ["seed", "sample_id", "date_id", "y_true"]].copy()
            one.loc[:, probability_cols] = probability
            one["method"] = method
            one["selected_l2"] = selected_l2 if method == "vector" else np.nan
            one["temperature"] = temperature.temperature_ if method == "temperature" else np.nan
            outputs.append(one)
    return pd.concat(outputs, ignore_index=True), pd.concat(receipts, ignore_index=True)


def run_i6_history_gate(
    uci_predictions: pd.DataFrame,
    tuning_records: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    primary_k: int,
    margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    convergence = rankings.loc[
        rankings["protocol"].eq("expanding_window")
        & rankings["method"].isin(["TCSS", "Pooled-GL"])
    ].copy()
    pooled_converged = pd.to_numeric(
        convergence.loc[convergence["method"].eq("Pooled-GL"), "converged_fraction"], errors="coerce"
    )
    gate_status = "VALID"
    if pooled_converged.empty or not bool((pooled_converged > 0).all()):
        gate_status = "INVALID_SELECTOR_FIT"

    tune = tuning_records.loc[
        tuning_records["protocol"].eq("expanding_window")
        & pd.to_numeric(tuning_records["K"], errors="coerce").eq(primary_k)
        & tuning_records["method"].isin(["TCSS", "Pooled-GL"])
    ].copy()
    best = tune.groupby(["seed", "origin_batch", "method"], as_index=False)["validation_macro_f1"].max()
    pivot = best.pivot(index=["seed", "origin_batch"], columns="method", values="validation_macro_f1")
    if not {"TCSS", "Pooled-GL"}.issubset(pivot.columns):
        raise ValueError("I6 lacks matched tuning records.")
    decisions: list[dict] = []
    output: list[pd.DataFrame] = []
    predictions = uci_predictions.loc[
        uci_predictions["protocol"].eq("expanding_window")
        & pd.to_numeric(uci_predictions["K"], errors="coerce").eq(primary_k)
        & uci_predictions["method"].isin(["TCSS", "Pooled-GL"])
    ].copy()
    for (seed_value, origin_value), row in pivot.iterrows():
        seed = int(seed_value)
        origin = int(origin_value)
        choose_tcss = gate_status == "VALID" and float(row["TCSS"]) > float(row["Pooled-GL"]) + margin
        chosen = "TCSS" if choose_tcss else "Pooled-GL"
        one = predictions.loc[
            predictions["seed"].eq(seed)
            & pd.to_numeric(predictions["origin_batch"], errors="coerce").eq(origin)
            & predictions["method"].eq(chosen)
        ].copy()
        if one.empty:
            raise ValueError(f"I6 lacks predictions for seed={seed}, origin={origin}, method={chosen}.")
        one["source_method"] = chosen
        one["method"] = "history_gated"
        output.append(one)
        decisions.append(
            {
                "seed": seed,
                "origin_batch": origin,
                "tcss_validation_macro_f1": float(row["TCSS"]),
                "pooled_validation_macro_f1": float(row["Pooled-GL"]),
                "margin": float(margin),
                "selected_method": chosen,
                "gate_status": gate_status,
                "test_labels_used_for_decision": False,
            }
        )
    return pd.concat(output, ignore_index=True), pd.DataFrame(decisions), gate_status


def balanced_training_availability(
    n_rows: int,
    group_names: Sequence[Hashable],
    *,
    random_counts: Sequence[int],
    seed: int,
    influence_ranking: Sequence[Hashable] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    names = tuple(group_names)
    n_groups = len(names)
    if n_rows < 1 or n_groups < 1:
        raise ValueError("Training rows and groups must be non-empty.")
    rng = np.random.default_rng(seed)
    masks: list[np.ndarray] = []
    records: list[dict] = []
    patterns: list[tuple[str, int]] = [("random", int(count)) for count in random_counts]
    if influence_ranking is not None:
        if set(influence_ranking) != set(names):
            raise ValueError("Influence ranking must contain every group exactly once.")
        patterns.extend(("influence", int(count)) for count in random_counts if int(count) > 0)
        ranked_indices = [names.index(group) for group in influence_ranking]
    else:
        ranked_indices = []
    for source_row in range(n_rows):
        for strategy, count in patterns:
            if count < 0 or count > n_groups:
                raise ValueError("Unavailable count lies outside the group range.")
            available = np.ones(n_groups, dtype=bool)
            if count:
                indices = (
                    rng.choice(n_groups, size=count, replace=False)
                    if strategy == "random"
                    else np.asarray(ranked_indices[:count], dtype=int)
                )
                available[indices] = False
            masks.append(available)
            records.append(
                {
                    "source_row": source_row,
                    "pattern_index": len(records),
                    "strategy": strategy,
                    "unavailable_count": count,
                }
            )
    return np.vstack(masks), pd.DataFrame(records)


def context_gated_representation(
    X: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    availability: np.ndarray,
    centers: Mapping[Hashable, np.ndarray],
) -> np.ndarray:
    imputed = apply_group_availability(X, groups, availability, centers, append_mask=False)
    available = np.asarray(availability, dtype=bool)
    if available.ndim == 1:
        available = np.repeat(available[None, :], len(imputed), axis=0)
    fraction = available.mean(axis=1, keepdims=True)
    return np.hstack([imputed, available.astype(float), fraction, imputed * fraction])


def expand_multimask_gated_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    *,
    random_counts: Sequence[int],
    seed: int,
    influence_ranking: Sequence[Hashable] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[Hashable, np.ndarray]]:
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    availability, receipt = balanced_training_availability(
        len(X),
        tuple(groups),
        random_counts=random_counts,
        seed=seed,
        influence_ranking=influence_ranking,
    )
    repeated_X = X[receipt["source_row"].to_numpy(dtype=int)]
    centers = group_centers(X, groups)
    features = context_gated_representation(repeated_X, groups, availability, centers)
    return features, y[receipt["source_row"].to_numpy(dtype=int)], receipt, centers


def run_i4_multimask_gated(
    X: np.ndarray,
    manifest: pd.DataFrame,
    groups: Mapping[Hashable, Sequence[int]],
    influence_ranking: Sequence[Hashable],
    *,
    run_id: str,
    seeds: Sequence[int],
    random_counts: Sequence[int],
    evaluation_counts: Sequence[int],
    strategies: Sequence[str],
    selected_k: int = 62,
    always_include_columns: Sequence[int] = (),
) -> tuple[list[PredictionBundle], pd.DataFrame, pd.DataFrame]:
    """Fit the v1.6 multi-mask gated-linear model and apply common stress masks."""

    from .baselines import tune_and_fit_probabilistic_model

    required = {"split", "label", "sample_id", "date"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"I4 manifest lacks {sorted(required - set(manifest.columns))}.")
    train = np.flatnonzero(manifest["split"].eq("train").to_numpy())
    validation = np.flatnonzero(manifest["split"].eq("validation").to_numpy())
    test = np.flatnonzero(manifest["split"].eq("test").to_numpy())
    selected = tuple(influence_ranking[:selected_k])
    columns, compact_groups = select_group_columns(dict(groups), selected)
    always_columns = np.asarray(tuple(int(value) for value in always_include_columns), dtype=int)
    if always_columns.size:
        columns = np.concatenate([columns, always_columns])
    selected_X = np.asarray(X[:, columns], dtype=float)
    transformer = RobustTrainTransformer().fit(selected_X[train])
    standardized = transformer.transform(selected_X)
    labels = manifest["label"].astype(str).to_numpy()
    sample_ids = manifest["sample_id"].to_numpy()
    dates = manifest["date"].to_numpy()
    bundles: list[PredictionBundle] = []
    training_receipts: list[pd.DataFrame] = []
    tuning_receipts: list[dict] = []

    for seed_value in seeds:
        seed = int(seed_value)
        expanded_X, expanded_y, training_receipt, centers = expand_multimask_gated_training(
            standardized[train],
            labels[train],
            compact_groups,
            random_counts=random_counts,
            seed=seed,
            influence_ranking=selected,
        )
        training_receipt.insert(0, "seed", seed)
        training_receipts.append(training_receipt)
        validation_availability = np.ones((len(validation), len(compact_groups)), dtype=bool)
        validation_X = context_gated_representation(
            standardized[validation], compact_groups, validation_availability, centers
        )
        model, tuning = tune_and_fit_probabilistic_model(
            "multinomial",
            expanded_X,
            expanded_y,
            validation_X,
            labels[validation],
            seed=seed,
        )
        tuning_receipts.extend({**row, "seed": seed} for row in tuning)
        for strategy in strategies:
            records = generate_evaluation_masks(
                selected,
                evaluation_counts,
                [seed],
                strategy=str(strategy),
                influence_ranking=selected if str(strategy) == "influence" else None,
            )
            for record in records:
                availability = availability_from_record(selected, record)
                features = context_gated_representation(
                    standardized[test], compact_groups, availability, centers
                )
                bundle = bundle_from_model(
                    model,
                    features,
                    labels[test],
                    run_id=run_id,
                    dataset="primary",
                    method="multimask_gated_linear",
                    split="test",
                    seed=seed,
                    sample_ids=sample_ids[test],
                    date_ids=dates[test],
                    metadata={
                        "mask_id": record.mask_id,
                        "mask_strategy": str(strategy),
                        "unavailable_count": int(record.unavailable_count),
                        "selected_k": int(selected_k),
                        "training_patterns_per_row": int(len(training_receipt) / len(train)),
                        "calibration": "none_classification_primary",
                    },
                )
                bundles.append(bundle)
    return (
        bundles,
        pd.concat(training_receipts, ignore_index=True),
        pd.DataFrame(tuning_receipts),
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(output_dir: Path, *, source_files: Sequence[Path], status: str) -> Path:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json")
    payload = {
        "status": status,
        "evidence_role": "POSTRESULT_EXPLORATORY",
        "source_files": {str(path): sha256(path) for path in source_files},
        "artifacts": {
            path.relative_to(output_dir).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
