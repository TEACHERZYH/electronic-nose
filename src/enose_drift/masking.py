"""Structured sensor masking shared across matched methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MaskRecord:
    mask_id: str
    strategy: str
    seed: int
    unavailable_count: int
    unavailable_groups: tuple[Hashable, ...]


def group_centers(X_train: np.ndarray, groups: Mapping[Hashable, Sequence[int]]) -> dict[Hashable, np.ndarray]:
    X_train = np.asarray(X_train, dtype=float)
    return {
        name: np.nanmedian(X_train[:, np.asarray(indices, dtype=int)], axis=0)
        for name, indices in groups.items()
    }


def apply_group_availability(
    X: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    available: np.ndarray,
    centers: Mapping[Hashable, np.ndarray],
    *,
    append_mask: bool = True,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    names = tuple(groups.keys())
    available = np.asarray(available, dtype=bool)
    if available.ndim == 1:
        available = np.repeat(available[None, :], len(X), axis=0)
    if available.shape != (len(X), len(names)):
        raise ValueError("Availability mask must have shape (n_rows, n_groups) or (n_groups,).")
    out = X.copy()
    for group_index, name in enumerate(names):
        columns = np.asarray(groups[name], dtype=int)
        missing_rows = ~available[:, group_index]
        out[np.ix_(missing_rows, columns)] = np.asarray(centers[name], dtype=float)
    return np.hstack([out, available.astype(float)]) if append_mask else out


def expand_structured_dropout_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    *,
    seed: int,
    max_fraction: float = 0.25,
    append_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[Hashable, np.ndarray]]:
    """Create one complete and one structured-dropout copy per training row."""

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    names = tuple(groups.keys())
    if not 0 <= max_fraction <= 1:
        raise ValueError("max_fraction must be between 0 and 1.")
    centers = group_centers(X_train, groups)
    n_groups = len(names)
    max_unavailable = int(np.ceil(max_fraction * n_groups))
    rng = np.random.default_rng(seed)
    complete = np.ones((len(X_train), n_groups), dtype=bool)
    dropout = np.ones_like(complete)
    for row in range(len(X_train)):
        unavailable_count = int(rng.integers(0, max_unavailable + 1))
        if unavailable_count:
            unavailable = rng.choice(n_groups, size=unavailable_count, replace=False)
            dropout[row, unavailable] = False
    full_features = apply_group_availability(X_train, groups, complete, centers, append_mask=append_mask)
    dropped_features = apply_group_availability(X_train, groups, dropout, centers, append_mask=append_mask)
    expanded_X = np.vstack([full_features, dropped_features])
    expanded_y = np.concatenate([y_train, y_train])
    expanded_mask = np.vstack([complete, dropout])
    return expanded_X, expanded_y, expanded_mask, centers


def generate_evaluation_masks(
    group_names: Sequence[Hashable],
    unavailable_counts: Sequence[int],
    seeds: Sequence[int],
    *,
    strategy: str,
    influence_ranking: Sequence[Hashable] | None = None,
) -> tuple[MaskRecord, ...]:
    names = tuple(group_names)
    name_set = set(names)
    if strategy not in {"random", "influence"}:
        raise ValueError("strategy must be random or influence.")
    if strategy == "influence":
        if influence_ranking is None or set(influence_ranking) != name_set:
            raise ValueError("Influence masks require a complete training-derived ranking.")
        ranked = tuple(influence_ranking)
    records: list[MaskRecord] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for count in unavailable_counts:
            if count < 0 or count > len(names):
                continue
            if strategy == "random":
                unavailable = tuple(rng.choice(names, size=count, replace=False).tolist()) if count else ()
            else:
                unavailable = ranked[:count]
            records.append(
                MaskRecord(
                    mask_id=f"{strategy}_seed{seed}_m{count}",
                    strategy=strategy,
                    seed=int(seed),
                    unavailable_count=int(count),
                    unavailable_groups=unavailable,
                )
            )
    return tuple(records)


def availability_from_record(group_names: Sequence[Hashable], record: MaskRecord) -> np.ndarray:
    unavailable = set(record.unavailable_groups)
    return np.asarray([name not in unavailable for name in group_names], dtype=bool)
