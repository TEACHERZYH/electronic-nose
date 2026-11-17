"""Stage-0 leakage and acquisition-protocol controls."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .baselines import tune_and_fit_probabilistic_model
from .evaluation import PredictionBundle, bundle_from_model


FORBIDDEN_RESPONSE_PATTERNS = (
    r"(^|_)file(name)?($|_)",
    r"(^|_)path($|_)",
    r"(^|_)row(_?id|_?number)?($|_)",
    r"(^|_)label(_?code)?($|_)",
    r"absolute.*time",
    r"future",
)


@dataclass(frozen=True)
class FeatureFamilies:
    full_response: np.ndarray
    baseline: np.ndarray
    temperature_humidity: np.ndarray
    recovery_history: np.ndarray


def identify_feature_families(feature_names: Sequence[str]) -> FeatureFamilies:
    names = [str(name) for name in feature_names]

    def matches(patterns: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [idx for idx, name in enumerate(names) if any(re.search(pattern, name, re.I) for pattern in patterns)],
            dtype=int,
        )

    baseline = matches((r"baseline", r"mean[ _-]*resistance", r"pre[ _-]*exposure"))
    th = matches((r"temperature", r"humidity", r"(^|_)T0?1", r"(^|_)H0?1"))
    history = matches((r"recovery", r"history", r"previous", r"carryover"))
    forbidden = set(matches(FORBIDDEN_RESPONSE_PATTERNS).tolist())
    positional = set(matches((r"ordinal", r"position", r"sequence", r"relative.*time")).tolist())
    # The response model deliberately excludes baseline and environmental
    # proxies; those appear only in their named Stage-0 controls.
    proxy_only = set(baseline.tolist()) | set(th.tolist())
    full = np.asarray(
        [idx for idx in range(len(names)) if idx not in forbidden | positional | proxy_only],
        dtype=int,
    )
    return FeatureFamilies(full, baseline, th, history)


def validate_response_feature_names(feature_names: Sequence[str]) -> None:
    violations = [
        name
        for name in map(str, feature_names)
        if any(re.search(pattern, name, re.I) for pattern in FORBIDDEN_RESPONSE_PATTERNS)
    ]
    if violations:
        raise ValueError(f"Forbidden response-model fields detected: {violations[:10]}")


def date_clustered_permute_labels(y: np.ndarray, dates: np.ndarray, seed: int) -> np.ndarray:
    """Permute labels within each complete date without moving rows across dates."""

    y = np.asarray(y).copy()
    dates = np.asarray(dates)
    rng = np.random.default_rng(seed)
    for date in np.unique(dates):
        rows = np.flatnonzero(dates == date)
        y[rows] = rng.permutation(y[rows])
    return y


def random_row_partition(
    y: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 24 / 39,
    validation_fraction: float = 7 / 39,
    required_test_groups: np.ndarray | None = None,
    minimum_per_required_group: int = 1,
    target_partition_sizes: tuple[int, int, int] | None = None,
):
    """Build the random-row diagnostic split.

    When ``required_test_groups`` is supplied, at least the requested number of
    rows from every non-empty group is reserved for the random-row test set.
    The remaining test quota and the validation split are stratified by class.
    This keeps the original split sizes while ensuring that the locked
    chronological test date-by-class cells are represented in every seed.
    """

    y = np.asarray(y)
    indices = np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_fraction, random_state=seed)
    train, remaining = next(splitter.split(indices, y))
    validation_share = validation_fraction / (1.0 - train_fraction)
    second = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - validation_share, random_state=seed + 1)
    validation_local, test_local = next(second.split(remaining, y[remaining]))
    if required_test_groups is None:
        return train, remaining[validation_local], remaining[test_local]

    groups = np.asarray(required_test_groups, dtype=object)
    if groups.shape != y.shape:
        raise ValueError("required_test_groups must have the same shape as y")
    if minimum_per_required_group < 1:
        raise ValueError("minimum_per_required_group must be positive")

    if target_partition_sizes is None:
        n_train = len(train)
        n_validation = len(validation_local)
        n_test = len(test_local)
    else:
        n_train, n_validation, n_test = map(int, target_partition_sizes)
        if min(n_train, n_validation, n_test) < 1 or n_train + n_validation + n_test != len(y):
            raise ValueError("target_partition_sizes must be three positive counts summing to len(y)")
    required_names = sorted(
        {str(value) for value in groups if value is not None and str(value).strip()}
    )
    rng = np.random.default_rng(seed)
    reserved: list[int] = []
    for name in required_names:
        candidates = np.flatnonzero(np.asarray([str(value) == name for value in groups]))
        if len(candidates) < minimum_per_required_group:
            raise ValueError(
                f"Required random-row test group {name!r} has only {len(candidates)} rows; "
                f"need {minimum_per_required_group}."
            )
        chosen = rng.choice(candidates, size=minimum_per_required_group, replace=False)
        reserved.extend(int(value) for value in np.asarray(chosen).tolist())

    reserved_array = np.asarray(sorted(set(reserved)), dtype=int)
    if len(reserved_array) > n_test:
        raise ValueError(
            f"Required random-row coverage needs {len(reserved_array)} test rows, but the frozen quota is {n_test}."
        )
    available = np.setdiff1d(indices, reserved_array, assume_unique=True)
    additional_count = n_test - len(reserved_array)
    if additional_count:
        additional_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=additional_count,
            random_state=seed + 2,
        )
        _, additional_local = next(additional_splitter.split(available, y[available]))
        test = np.concatenate([reserved_array, available[additional_local]])
    else:
        test = reserved_array

    remaining_after_test = np.setdiff1d(indices, test, assume_unique=False)
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_validation,
        random_state=seed + 3,
    )
    train_local, validation_local = next(
        validation_splitter.split(remaining_after_test, y[remaining_after_test])
    )
    train = remaining_after_test[train_local]
    validation = remaining_after_test[validation_local]

    covered = {str(value) for value in groups[test] if value is not None and str(value).strip()}
    if covered != set(required_names):
        raise RuntimeError("Constrained random-row split did not preserve every required test group.")
    if len(train) + len(validation) + len(test) != len(y):
        raise RuntimeError("Constrained random-row split lost or duplicated rows.")
    if (len(train), len(validation), len(test)) != (n_train, n_validation, n_test):
        raise RuntimeError("Constrained random-row split did not preserve the frozen partition sizes.")
    return np.sort(train), np.sort(validation), np.sort(test)


def _ordinal_probability(
    train_ordinal: np.ndarray,
    train_y: np.ndarray,
    target_ordinal: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    probability = np.empty((len(target_ordinal), len(classes)), dtype=float)
    global_counts = np.asarray([(train_y == label).sum() for label in classes], dtype=float) + 1.0
    for row, ordinal in enumerate(target_ordinal):
        observed = train_y[train_ordinal == ordinal]
        counts = np.asarray([(observed == label).sum() for label in classes], dtype=float) + 1.0
        if not len(observed):
            counts = global_counts.copy()
        probability[row] = counts / counts.sum()
    return probability


def run_ordinal_control(
    manifest,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    run_id: str,
    seed: int,
    dataset: str = "primary",
) -> PredictionBundle:
    label_col = "label" if "label" in manifest.columns else "class"
    order_col = "observed_ordinal" if "observed_ordinal" in manifest.columns else "within_day_order"
    required = {"sample_id", "date", label_col, order_col}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Ordinal control requires manifest columns {sorted(required)}")
    classes = np.asarray(sorted(manifest[label_col].unique().tolist()))
    probability = _ordinal_probability(
        manifest.iloc[train_indices][order_col].to_numpy(),
        manifest.iloc[train_indices][label_col].to_numpy(),
        manifest.iloc[target_indices][order_col].to_numpy(),
        classes,
    )
    bundle = PredictionBundle(
        run_id=run_id,
        dataset=dataset,
        method="observed_ordinal",
        split="test",
        seed=seed,
        sample_ids=manifest.iloc[target_indices]["sample_id"].to_numpy(),
        date_ids=manifest.iloc[target_indices]["date"].to_numpy(),
        y_true=manifest.iloc[target_indices][label_col].to_numpy(),
        classes=classes,
        probabilities=probability,
        logits=np.log(np.clip(probability, 1e-15, 1.0)),
        metadata={"access": "acquisition_control_only", "not_a_response_model": True},
    )
    bundle.validate()
    return bundle


def run_feature_control(
    control_name: str,
    X: np.ndarray,
    manifest,
    feature_names: Sequence[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    run_id: str,
    seed: int,
    model_name: str = "rbf_svm",
    permute_training_labels: bool = False,
) -> tuple[PredictionBundle, PredictionBundle, list[dict]]:
    families = identify_feature_families(feature_names)
    control_to_columns = {
        "full_response": families.full_response,
        "baseline_only": families.baseline,
        "temperature_humidity_only": families.temperature_humidity,
        "recovery_history_only": families.recovery_history,
        "clustered_permutation": families.full_response,
        "random_row": families.full_response,
        "chronological_forward": families.full_response,
    }
    if control_name not in control_to_columns:
        raise ValueError(f"Unsupported feature control {control_name!r}.")
    columns = control_to_columns[control_name]
    if not len(columns):
        raise ValueError(f"No feature columns were found for {control_name}.")
    label_col = "label" if "label" in manifest.columns else "class"
    y = manifest[label_col].to_numpy()
    dates = manifest["date"].to_numpy()
    y_train = y[train_indices]
    if permute_training_labels:
        y_train = date_clustered_permute_labels(y_train, dates[train_indices], seed)
    model, tuning = tune_and_fit_probabilistic_model(
        model_name,
        X[np.ix_(train_indices, columns)],
        y_train,
        X[np.ix_(validation_indices, columns)],
        y[validation_indices],
        seed=seed,
    )
    validation = bundle_from_model(
        model,
        X[np.ix_(validation_indices, columns)],
        y[validation_indices],
        run_id=run_id,
        dataset="primary",
        method=control_name,
        split="validation",
        seed=seed,
        sample_ids=manifest.iloc[validation_indices]["sample_id"].to_numpy(),
        date_ids=dates[validation_indices],
        metadata={"feature_count": int(len(columns)), "tuning": tuning},
    )
    test = bundle_from_model(
        model,
        X[np.ix_(test_indices, columns)],
        y[test_indices],
        run_id=run_id,
        dataset="primary",
        method=control_name,
        split="test",
        seed=seed,
        sample_ids=manifest.iloc[test_indices]["sample_id"].to_numpy(),
        date_ids=dates[test_indices],
        metadata={"feature_count": int(len(columns)), "tuning": tuning},
    )
    return validation, test, tuning
