"""Classification, calibration, and selective-risk metrics.

The functions in this module operate on real prediction arrays and contain no
simulation-specific constants.  Probability columns always follow the explicit
``labels`` order.  When ``labels`` is omitted, probability metrics require
integer labels ``0, ..., C - 1`` so that a missing class cannot silently change
the column mapping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def _as_1d(values: Iterable[Any] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _normalise_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray | None,
    labels: Sequence[Any] | None,
) -> tuple[Any, ...]:
    if labels is None:
        combined = y_true if y_pred is None else np.concatenate((y_true, y_pred))
        try:
            resolved = tuple(np.unique(combined).tolist())
        except TypeError as exc:
            raise ValueError("labels must be supplied when class values cannot be sorted") from exc
    else:
        resolved = tuple(labels)
    if not resolved:
        raise ValueError("labels must not be empty")
    try:
        if len(set(resolved)) != len(resolved):
            raise ValueError("labels must be unique")
    except TypeError as exc:
        raise ValueError("class labels must be hashable") from exc
    return resolved


def confusion_matrix(
    y_true: Iterable[Any] | np.ndarray,
    y_pred: Iterable[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
) -> np.ndarray:
    """Return a multiclass confusion matrix with true classes on rows."""

    truth = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_pred, "y_pred")
    if truth.size != prediction.size:
        raise ValueError("y_true and y_pred must contain the same number of samples")
    resolved = _normalise_labels(truth, prediction, labels)
    index = {label: class_id for class_id, label in enumerate(resolved)}
    try:
        true_index = np.fromiter((index[value] for value in truth), dtype=int, count=truth.size)
        pred_index = np.fromiter((index[value] for value in prediction), dtype=int, count=prediction.size)
    except KeyError as exc:
        raise ValueError(f"observed class {exc.args[0]!r} is absent from labels") from exc
    matrix = np.zeros((len(resolved), len(resolved)), dtype=np.int64)
    np.add.at(matrix, (true_index, pred_index), 1)
    return matrix


def metrics_from_confusion(
    matrix: np.ndarray,
    *,
    supported_only: bool = False,
    zero_division: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """Compute recognition metrics from a square multiclass confusion matrix.

    ``supported_only=True`` excludes classes with no true samples from macro-F1
    and balanced accuracy.  This is the intended mode for UCI
    supported-class reporting.  The default keeps every declared class, which
    is the conservative fixed-class convention.
    """

    counts = np.asarray(matrix, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square array")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("matrix entries must be finite and nonnegative")
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("matrix must contain at least one observation")

    true_total = counts.sum(axis=1)
    pred_total = counts.sum(axis=0)
    true_positive = np.diag(counts)
    recall = np.divide(
        true_positive,
        true_total,
        out=np.full_like(true_positive, float(zero_division)),
        where=true_total > 0,
    )
    precision = np.divide(
        true_positive,
        pred_total,
        out=np.full_like(true_positive, float(zero_division)),
        where=pred_total > 0,
    )
    f1_denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.full_like(true_positive, float(zero_division)),
        where=f1_denominator > 0,
    )

    included = true_total > 0 if supported_only else np.ones(counts.shape[0], dtype=bool)
    if not np.any(included):
        raise ValueError("no supported class is available for macro averaging")

    correct = float(true_positive.sum())
    first_term = max(total**2 - float(np.dot(pred_total, pred_total)), 0.0)
    second_term = max(total**2 - float(np.dot(true_total, true_total)), 0.0)
    denominator = np.sqrt(first_term * second_term)
    mcc = 0.0 if denominator == 0 else (correct * total - float(np.dot(pred_total, true_total))) / denominator

    return {
        "macro_f1": float(np.mean(f1[included])),
        "balanced_accuracy": float(np.mean(recall[included])),
        "mcc": float(mcc),
        "accuracy": float(correct / total),
        "class_recall": recall,
        "class_precision": precision,
        "class_f1": f1,
        "class_support": true_total.astype(np.int64),
    }


def classification_metrics(
    y_true: Iterable[Any] | np.ndarray,
    y_pred: Iterable[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    supported_only: bool = False,
    zero_division: float = 0.0,
) -> dict[str, Any]:
    """Compute a confusion matrix and all recognition metrics in one call."""

    truth = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_pred, "y_pred")
    resolved = _normalise_labels(truth, prediction, labels)
    matrix = confusion_matrix(truth, prediction, labels=resolved)
    result: dict[str, Any] = {
        "labels": list(resolved),
        "confusion_matrix": matrix,
    }
    result.update(
        metrics_from_confusion(
            matrix,
            supported_only=supported_only,
            zero_division=zero_division,
        )
    )
    return result


def _probability_inputs(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    labels: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    truth = _as_1d(y_true, "y_true")
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[0] != truth.size or probability.shape[1] == 0:
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes) aligned with y_true"
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("probabilities must be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    row_sums = probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("each probability row must sum to one")

    if labels is None:
        resolved: tuple[Any, ...] = tuple(range(probability.shape[1]))
    else:
        resolved = tuple(labels)
        if len(resolved) != probability.shape[1]:
            raise ValueError("labels length must equal the number of probability columns")
        if len(set(resolved)) != len(resolved):
            raise ValueError("labels must be unique")
    class_index = {label: class_id for class_id, label in enumerate(resolved)}
    try:
        true_index = np.fromiter((class_index[value] for value in truth), dtype=int, count=truth.size)
    except KeyError as exc:
        raise ValueError(f"observed class {exc.args[0]!r} is absent from labels") from exc
    return true_index, probability, resolved


def multiclass_brier_score(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """Return the unscaled multiclass Brier score."""

    true_index, probability, _ = _probability_inputs(y_true, probabilities, labels)
    one_hot = np.eye(probability.shape[1], dtype=float)[true_index]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def negative_log_likelihood(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Return mean multiclass negative log-likelihood."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    true_index, probability, _ = _probability_inputs(y_true, probabilities, labels)
    true_probability = probability[np.arange(true_index.size), true_index]
    return float(-np.mean(np.log(np.clip(true_probability, epsilon, 1.0))))


def expected_calibration_error(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    n_bins: int = 10,
) -> float:
    """Compute top-label ECE with equal-width confidence bins."""

    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    true_index, probability, _ = _probability_inputs(y_true, probabilities, labels)
    predicted = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = predicted == true_index
    bin_index = np.minimum((confidence * int(n_bins)).astype(int), int(n_bins) - 1)
    ece = 0.0
    for bin_id in range(int(n_bins)):
        selected = bin_index == bin_id
        if np.any(selected):
            weight = float(np.mean(selected))
            ece += weight * abs(float(np.mean(correct[selected])) - float(np.mean(confidence[selected])))
    return float(ece)


def risk_coverage_from_confidence(
    correct: Iterable[bool] | np.ndarray,
    confidence: Iterable[float] | np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Electronic-nose numerical routines."""

    correctness = _as_1d(correct, "correct").astype(bool, copy=False)
    certainty = _as_1d(confidence, "confidence").astype(float, copy=False)
    if correctness.size != certainty.size:
        raise ValueError("correct and confidence must have equal length")
    if not np.all(np.isfinite(certainty)) or np.any(certainty < 0.0) or np.any(certainty > 1.0):
        raise ValueError("confidence values must be finite and lie in [0, 1]")
    order = np.argsort(-certainty, kind="stable")
    ordered_errors = 1.0 - correctness[order].astype(float)
    coverage = np.arange(1, correctness.size + 1, dtype=float) / correctness.size
    risk = np.cumsum(ordered_errors) / np.arange(1, correctness.size + 1, dtype=float)
    if coverage.size < 2:
        aurc = 0.0
    elif hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risk, coverage))
    else:  # NumPy < 2.0 compatibility.
        aurc = float(np.trapz(risk, coverage))
    return {
        "coverage": coverage,
        "risk": risk,
        "confidence": certainty[order],
        "order": order,
        "aurc": aurc,
        "full_coverage_risk": float(risk[-1]),
    }


def risk_coverage_curve(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute risk-coverage data directly from multiclass probabilities."""

    true_index, probability, _ = _probability_inputs(y_true, probabilities, labels)
    predicted = probability.argmax(axis=1)
    return risk_coverage_from_confidence(predicted == true_index, probability.max(axis=1))


def probability_metrics(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Brier, NLL, ECE, and complete risk-coverage outputs."""

    curve = risk_coverage_curve(y_true, probabilities, labels=labels)
    half_index = max(0, int(np.ceil(0.50 * len(curve["coverage"]))) - 1)
    return {
        "brier": multiclass_brier_score(y_true, probabilities, labels=labels),
        "nll": negative_log_likelihood(y_true, probabilities, labels=labels),
        "ece": expected_calibration_error(y_true, probabilities, labels=labels, n_bins=n_bins),
        "risk_at_50pct_coverage": float(curve["risk"][half_index]),
        **curve,
    }


def date_level_metrics(
    y_true: Iterable[Any] | np.ndarray,
    y_pred: Iterable[Any] | np.ndarray,
    date_ids: Iterable[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    probabilities: np.ndarray | None = None,
    supported_only: bool = False,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Aggregate sample predictions into one metric record per date.

    Dates retain first-observed order.  A common global class order is used for
    every date, preventing absent classes from changing the macro denominator.
    """

    truth = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_pred, "y_pred")
    dates = _as_1d(date_ids, "date_ids")
    if not (truth.size == prediction.size == dates.size):
        raise ValueError("y_true, y_pred, and date_ids must have equal length")
    resolved = _normalise_labels(truth, prediction, labels)
    probability: np.ndarray | None = None
    if probabilities is not None:
        _, probability, probability_labels = _probability_inputs(truth, probabilities, resolved)
        if probability_labels != resolved:
            raise ValueError("probability labels do not match classification labels")

    ordered_dates: list[Any] = []
    seen: set[Any] = set()
    for date_id in dates.tolist():
        if date_id not in seen:
            seen.add(date_id)
            ordered_dates.append(date_id)

    records: list[dict[str, Any]] = []
    for date_id in ordered_dates:
        selected = dates == date_id
        metrics = classification_metrics(
            truth[selected],
            prediction[selected],
            labels=resolved,
            supported_only=supported_only,
        )
        record: dict[str, Any] = {
            "date_id": date_id,
            "n": int(np.sum(selected)),
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "mcc": metrics["mcc"],
            "accuracy": metrics["accuracy"],
            "labels": list(resolved),
            "class_recall": metrics["class_recall"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
        if probability is not None:
            record.update(
                probability_metrics(
                    truth[selected],
                    probability[selected],
                    labels=resolved,
                    n_bins=n_bins,
                )
            )
        records.append(record)
    return records



macro_classification_metrics = classification_metrics


def area_under_risk_coverage(
    y_true: Iterable[Any] | np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """Return AURC while retaining the positive-coverage integration rule."""

    return float(risk_coverage_curve(y_true, probabilities, labels=labels)["aurc"])


__all__ = [
    "area_under_risk_coverage",
    "classification_metrics",
    "confusion_matrix",
    "date_level_metrics",
    "expected_calibration_error",
    "macro_classification_metrics",
    "metrics_from_confusion",
    "multiclass_brier_score",
    "negative_log_likelihood",
    "probability_metrics",
    "risk_coverage_curve",
    "risk_coverage_from_confidence",
]
