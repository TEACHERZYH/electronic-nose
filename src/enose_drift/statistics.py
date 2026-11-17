"""Date-level aggregation and inferential statistics for the experiment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_BOOTSTRAP_SEED = 20260826


def _finite_1d(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def aggregate_seed_metrics_by_date(
    rows: Iterable[Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    date_key: str = "date_id",
    seed_key: str = "seed",
    group_keys: Sequence[str] = (),
    expected_seeds: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Average seed-level metric records within each date and method setting.

    Every ``group_keys + date + seed`` combination must occur exactly once.
    This catches accidental row duplication before the date is treated as the
    scientific resampling unit.
    """

    metrics = tuple(metric_names)
    groups = tuple(group_keys)
    if not metrics:
        raise ValueError("metric_names must not be empty")
    if len(set(metrics)) != len(metrics) or len(set(groups)) != len(groups):
        raise ValueError("metric_names and group_keys must not contain duplicates")
    if date_key in groups or seed_key in groups:
        raise ValueError("date_key and seed_key must not also appear in group_keys")
    expected_sequence = None if expected_seeds is None else tuple(expected_seeds)
    expected = None if expected_sequence is None else set(expected_sequence)
    if expected is not None and len(expected) != len(expected_sequence):
        raise ValueError("expected_seeds must be unique")

    grouped: dict[tuple[Any, ...], dict[Any, Mapping[str, Any]]] = defaultdict(dict)
    first_order: dict[tuple[Any, ...], int] = {}
    for row_index, row in enumerate(rows):
        required = (*groups, date_key, seed_key, *metrics)
        missing = [key for key in required if key not in row]
        if missing:
            raise KeyError(f"row {row_index} is missing required fields: {missing}")
        key = tuple(row[field] for field in groups) + (row[date_key],)
        seed = row[seed_key]
        if seed in grouped[key]:
            raise ValueError(f"duplicate seed record for group/date {key!r}, seed {seed!r}")
        grouped[key][seed] = row
        first_order.setdefault(key, row_index)

    if not grouped:
        raise ValueError("rows must contain at least one record")

    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=first_order.__getitem__):
        seed_rows = grouped[key]
        observed_seeds = set(seed_rows)
        if expected is not None and observed_seeds != expected:
            missing = sorted(expected - observed_seeds, key=repr)
            extra = sorted(observed_seeds - expected, key=repr)
            raise ValueError(f"seed mismatch for group/date {key!r}: missing={missing}, extra={extra}")
        record = {field: key[index] for index, field in enumerate((*groups, date_key))}
        record["n_seeds"] = len(seed_rows)
        for metric in metrics:
            values = _finite_1d(
                [seed_rows[seed][metric] for seed in seed_rows],
                f"{metric} for {key!r}",
            )
            record[metric] = float(np.mean(values))
        output.append(record)
    return output


def paired_date_cluster_bootstrap(
    values_a: Iterable[float] | np.ndarray,
    values_b: Iterable[float] | np.ndarray,
    *,
    dates: Sequence[Any] | None = None,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the mean paired difference using complete dates as clusters.

    Inputs must already contain one value per date, normally after averaging
    the five fitting seeds.  The estimand is ``mean(values_a - values_b)``.
    """

    first = _finite_1d(values_a, "values_a")
    second = _finite_1d(values_b, "values_b")
    if first.size != second.size:
        raise ValueError("values_a and values_b must have equal length")
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if dates is None:
        date_values: list[Any] = list(range(first.size))
    else:
        date_values = list(dates)
        if len(date_values) != first.size:
            raise ValueError("dates must align with values_a and values_b")
        try:
            unique_dates = set(date_values)
        except TypeError as exc:
            raise ValueError("date identifiers must be hashable") from exc
        if len(unique_dates) != len(date_values):
            raise ValueError("inputs must contain exactly one aggregate value per date")

    differences = first - second
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        differences.size,
        size=(int(n_resamples), differences.size),
    )
    sampled_means = differences[sampled_indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(sampled_means, [tail, 1.0 - tail])
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": float(confidence_level),
        "n_dates": int(differences.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "dates": date_values,
    }


def paired_date_cluster_bootstrap_from_mappings(
    values_a: Mapping[Any, float],
    values_b: Mapping[Any, float],
    **kwargs: Any,
) -> dict[str, Any]:
    """Align two date-score mappings before paired cluster bootstrap."""

    dates_a = list(values_a)
    if set(dates_a) != set(values_b):
        missing_from_a = sorted(set(values_b) - set(values_a), key=repr)
        missing_from_b = sorted(set(values_a) - set(values_b), key=repr)
        raise ValueError(
            f"date keys differ: missing_from_a={missing_from_a}, missing_from_b={missing_from_b}"
        )
    return paired_date_cluster_bootstrap(
        [values_a[date] for date in dates_a],
        [values_b[date] for date in dates_a],
        dates=dates_a,
        **kwargs,
    )


def holm_adjust(p_values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return family-wise Holm-adjusted p-values in the original order."""

    values = _finite_1d(p_values, "p_values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    multiplier = np.arange(values.size, 0, -1, dtype=float)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(multiplier * sorted_values))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def holm_adjust_named(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm correction while preserving comparison names and order."""

    names = list(p_values)
    if not names:
        raise ValueError("p_values must not be empty")
    adjusted = holm_adjust([p_values[name] for name in names])
    return {name: float(value) for name, value in zip(names, adjusted)}


def summarize_date_scores(
    date_ids: Sequence[Any],
    scores: Iterable[float] | np.ndarray,
    *,
    elapsed_time: Iterable[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize chronologically ordered date scores without row weighting."""

    values = _finite_1d(scores, "scores")
    dates = list(date_ids)
    if len(dates) != values.size:
        raise ValueError("date_ids and scores must have equal length")
    if len(set(dates)) != len(dates):
        raise ValueError("date_ids must be unique")
    worst_index = int(np.argmin(values))
    slope: float | None = None
    if elapsed_time is not None:
        elapsed = _finite_1d(elapsed_time, "elapsed_time")
        if elapsed.size != values.size:
            raise ValueError("elapsed_time and scores must have equal length")
        if values.size < 2 or np.ptp(elapsed) <= 0:
            raise ValueError("elapsed_time must contain at least two distinct values")
        slope = float(np.polyfit(elapsed, values, 1)[0])
    return {
        "n_dates": int(values.size),
        "mean": float(np.mean(values)),
        "worst": float(values[worst_index]),
        "worst_date": dates[worst_index],
        "final": float(values[-1]),
        "final_date": dates[-1],
        "slope_per_time_unit": slope,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "aggregate_seed_metrics_by_date",
    "holm_adjust",
    "holm_adjust_named",
    "paired_date_cluster_bootstrap",
    "paired_date_cluster_bootstrap_from_mappings",
    "summarize_date_scores",
]
