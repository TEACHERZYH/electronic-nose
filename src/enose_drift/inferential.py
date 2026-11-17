"""Frozen date-level exploratory comparisons and Holm adjustment."""

from __future__ import annotations

from itertools import product
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .statistics import aggregate_seed_metrics_by_date, holm_adjust_named


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _aggregate(frame: pd.DataFrame, expected_seeds: Sequence[int]) -> dict[str, float]:
    records = aggregate_seed_metrics_by_date(
        frame.to_dict(orient="records"),
        ("macro_f1",),
        date_key="date_id",
        seed_key="seed",
        expected_seeds=expected_seeds,
    )
    return {str(row["date_id"]): float(row["macro_f1"]) for row in records}


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    observed = abs(float(np.mean(values)))
    permuted = np.asarray(
        [np.mean(values * np.asarray(signs, dtype=float)) for signs in product((-1.0, 1.0), repeat=len(values))]
    )
    return float(np.sum(np.abs(permuted) >= observed - 1e-15) / len(permuted))


def exploratory_holm_comparisons(
    resource_date_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = (11, 23, 37, 53, 71),
    expected_dates: int = 8,
) -> pd.DataFrame:
    """Compare the primary TCSS setting with frozen exploratory baselines."""

    frame = resource_date_metrics.copy()
    if "split" in frame:
        frame = frame.loc[frame["split"].astype(str).str.casefold().eq("test")]
    frame["_method"] = frame["method"].map(_key)
    k = pd.to_numeric(frame.get("sensor_budget"), errors="coerce")
    budget = pd.to_numeric(frame.get("blank_budget"), errors="coerce")
    reference_rows = frame.loc[
        frame["_method"].eq("tcss") & k.eq(16) & budget.eq(1)
    ]
    reference = _aggregate(reference_rows, expected_seeds)
    if len(reference) != expected_dates:
        raise ValueError(f"Expected {expected_dates} TCSS test dates, found {len(reference)}")

    candidate_specs = {
        "pooled_group_lasso_K16_B1": frame["_method"].eq("pooled_group_lasso") & k.eq(16) & budget.eq(1),
        "mRMR_SVM_K16_B1": frame["_method"].eq("mrmr_svm") & k.eq(16) & budget.eq(1),
        "LDA_source_only": frame["_method"].eq("lda"),
        "RBF_SVM_source_only": frame["_method"].eq("rbf_svm"),
        "CORAL_SVM_unlabeled_target": frame["_method"].eq("coral_svm"),
    }
    rows: list[dict[str, object]] = []
    raw_p: dict[str, float] = {}
    for name, selector in candidate_specs.items():
        candidate_rows = frame.loc[selector]
        if candidate_rows.empty:
            continue
        candidate = _aggregate(candidate_rows, expected_seeds)
        if set(candidate) != set(reference):
            raise ValueError(f"{name} does not contain the same test dates as TCSS")
        dates = list(reference)
        differences = np.asarray([reference[date] - candidate[date] for date in dates], dtype=float)
        p_value = _exact_sign_flip_pvalue(differences)
        raw_p[name] = p_value
        rows.append(
            {
                "comparison": name,
                "estimand": "TCSS_K16_B1_minus_baseline_date_macro_f1",
                "n_dates": len(dates),
                "mean_difference": float(np.mean(differences)),
                "raw_two_sided_sign_flip_p": p_value,
                "information_access_note": (
                    "matched_sparse"
                    if "K16_B1" in name
                    else ("unlabeled_locked_test" if name.startswith("CORAL") else "source_only_full_array")
                ),
            }
        )
    adjusted = holm_adjust_named(raw_p) if raw_p else {}
    for row in rows:
        row["holm_adjusted_p"] = adjusted[str(row["comparison"])]
    return pd.DataFrame(rows)


__all__ = ["exploratory_holm_comparisons"]
