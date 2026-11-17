"""Prediction records, date-level metrics, calibration, and paired comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .calibration import TemperatureScaler
from .metrics import date_level_metrics, probability_metrics
from .statistics import paired_date_cluster_bootstrap_from_mappings, summarize_date_scores


@dataclass(frozen=True)
class PredictionBundle:
    run_id: str
    dataset: str
    method: str
    split: str
    seed: int
    sample_ids: np.ndarray
    date_ids: np.ndarray
    y_true: np.ndarray
    classes: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        n = len(self.y_true)
        if any(len(values) != n for values in (self.sample_ids, self.date_ids, self.probabilities)):
            raise ValueError("Prediction arrays have inconsistent row counts.")
        if self.probabilities.shape != (n, len(self.classes)):
            raise ValueError("Probability matrix has an invalid class dimension.")
        if not np.isfinite(self.probabilities).all() or not np.allclose(
            self.probabilities.sum(axis=1), 1.0, atol=1e-8
        ):
            raise ValueError("Probabilities must be finite and sum to one.")
        if self.logits is not None and self.logits.shape != self.probabilities.shape:
            raise ValueError("Logits and probabilities must have the same shape.")

    @property
    def y_pred(self) -> np.ndarray:
        self.validate()
        return self.classes[np.argmax(self.probabilities, axis=1)]

    def sample_frame(self) -> pd.DataFrame:
        self.validate()
        frame = pd.DataFrame(
            {
                "run_id": self.run_id,
                "dataset": self.dataset,
                "method": self.method,
                "split": self.split,
                "seed": self.seed,
                "sample_id": self.sample_ids,
                "date_id": self.date_ids,
                "y_true": self.y_true,
                "y_pred": self.y_pred,
            }
        )
        for class_index, label in enumerate(self.classes):
            frame[f"p_{label}"] = self.probabilities[:, class_index]
        return frame

    def date_records(self, *, supported_only: bool = False) -> list[dict[str, Any]]:
        records = date_level_metrics(
            self.y_true,
            self.y_pred,
            self.date_ids,
            labels=self.classes.tolist(),
            probabilities=self.probabilities,
            supported_only=supported_only,
            n_bins=10,
        )
        for record in records:
            record.update(
                {
                    "run_id": self.run_id,
                    "dataset": self.dataset,
                    "method": self.method,
                    "split": self.split,
                    "seed": self.seed,
                }
            )
        return records


def bundle_from_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    run_id: str,
    dataset: str,
    method: str,
    split: str,
    seed: int,
    sample_ids: np.ndarray,
    date_ids: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> PredictionBundle:
    probability = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    if hasattr(model, "decision_function"):
        logits = np.asarray(model.decision_function(X), dtype=float)
        if logits.ndim == 1:
            logits = np.column_stack([-logits, logits])
        # Pairwise SVC decision values do not have one column per class.
        if logits.shape != probability.shape:
            logits = np.log(np.clip(probability, 1e-15, 1.0))
    else:
        logits = np.log(np.clip(probability, 1e-15, 1.0))
    bundle = PredictionBundle(
        run_id=run_id,
        dataset=dataset,
        method=method,
        split=split,
        seed=int(seed),
        sample_ids=np.asarray(sample_ids),
        date_ids=np.asarray(date_ids),
        y_true=np.asarray(y),
        classes=classes,
        probabilities=probability,
        logits=logits,
        metadata=metadata,
    )
    bundle.validate()
    return bundle


def calibrate_bundle(validation: PredictionBundle, target: PredictionBundle) -> tuple[PredictionBundle, dict]:
    validation.validate()
    target.validate()
    if validation.classes.tolist() != target.classes.tolist():
        raise ValueError("Validation and target bundles use different class orders.")
    if validation.logits is None or target.logits is None:
        raise ValueError("Temperature scaling requires validation and target logits.")
    scaler = TemperatureScaler().fit(validation.logits, validation.y_true, validation.classes)
    calibrated = PredictionBundle(
        run_id=target.run_id,
        dataset=target.dataset,
        method=f"{target.method}+temperature",
        split=target.split,
        seed=target.seed,
        sample_ids=target.sample_ids,
        date_ids=target.date_ids,
        y_true=target.y_true,
        classes=target.classes,
        probabilities=scaler.predict_proba(target.logits),
        logits=target.logits,
        metadata={**(target.metadata or {}), "calibration": scaler.audit_record()},
    )
    calibrated.validate()
    return calibrated, scaler.audit_record()


def overall_probability_record(bundle: PredictionBundle, *, fixed_coverage: float = 0.50) -> dict[str, Any]:
    record = probability_metrics(bundle.y_true, bundle.probabilities, labels=bundle.classes.tolist(), n_bins=10)
    confidence = np.max(bundle.probabilities, axis=1)
    correct = bundle.y_pred == bundle.y_true
    keep = max(1, int(np.ceil(fixed_coverage * len(confidence))))
    order = np.argsort(-confidence, kind="stable")[:keep]
    record["fixed_coverage"] = float(keep / len(confidence))
    record["fixed_coverage_risk"] = float(1.0 - np.mean(correct[order]))
    return record


def compare_date_macro_f1(
    records_a: Sequence[dict],
    records_b: Sequence[dict],
    *,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 11,
) -> dict[str, Any]:
    mapping_a = {str(record["date_id"]): float(record["macro_f1"]) for record in records_a}
    mapping_b = {str(record["date_id"]): float(record["macro_f1"]) for record in records_b}
    result = paired_date_cluster_bootstrap_from_mappings(
        mapping_a,
        mapping_b,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    ordered_dates = list(mapping_a.keys())
    result["summary_a"] = summarize_date_scores(ordered_dates, [mapping_a[date] for date in ordered_dates])
    result["summary_b"] = summarize_date_scores(ordered_dates, [mapping_b[date] for date in ordered_dates])
    return result
