"""Causal blank anchoring under fixed B=0/1/3 information budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AnchorUse:
    row_index: int
    date: str
    budget: int
    effective_budget: int
    anchor_indices: tuple[int, ...]
    latest_anchor_order: float | None
    scored_order: float


class BlankAnchorAdjuster:
    def __init__(self, groups: Mapping[Hashable, Sequence[int]]) -> None:
        self.groups = {name: np.asarray(indices, dtype=int) for name, indices in groups.items()}

    def fit(self, X_train: np.ndarray, train_anchor_mask: np.ndarray) -> "BlankAnchorAdjuster":
        X_train = np.asarray(X_train, dtype=float)
        train_anchor_mask = np.asarray(train_anchor_mask, dtype=bool)
        if X_train.ndim != 2 or train_anchor_mask.shape != (len(X_train),):
            raise ValueError("Training matrix and anchor mask shapes are inconsistent.")
        if not np.any(train_anchor_mask):
            raise ValueError("At least one training blank anchor is required.")
        self.source_anchor_ = np.nanmedian(X_train[train_anchor_mask], axis=0)
        if not np.isfinite(self.source_anchor_).all():
            raise ValueError("Training blank anchor contains non-finite feature centers.")
        self.n_features_in_ = X_train.shape[1]
        return self

    def transform(
        self,
        X: np.ndarray,
        dates: np.ndarray,
        within_date_order: np.ndarray,
        candidate_anchor_mask: np.ndarray,
        scored_mask: np.ndarray,
        *,
        budget: int,
        allow_partial_budget: bool = False,
    ) -> tuple[np.ndarray, tuple[AnchorUse, ...]]:
        if not hasattr(self, "source_anchor_"):
            raise RuntimeError("BlankAnchorAdjuster must be fitted on training anchors.")
        if budget not in {0, 1, 3}:
            raise ValueError("The frozen blank budget must be one of 0, 1 or 3.")
        X = np.asarray(X, dtype=float)
        dates = np.asarray(dates)
        order = np.asarray(within_date_order, dtype=float)
        candidate = np.asarray(candidate_anchor_mask, dtype=bool)
        scored = np.asarray(scored_mask, dtype=bool)
        n = len(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature matrix differs from the fitted anchor schema.")
        if any(vector.shape != (n,) for vector in (dates, order, candidate, scored)):
            raise ValueError("Metadata vectors must align with X.")
        if np.any(candidate & scored):
            raise ValueError("Candidate blank anchors must be excluded from every scoring set.")

        adjusted = X.copy()
        receipt: list[AnchorUse] = []
        for row in np.flatnonzero(scored):
            allowed = np.flatnonzero(candidate & (dates == dates[row]) & (order < order[row]))
            allowed = allowed[np.argsort(order[allowed], kind="stable")]
            if budget == 0:
                chosen = np.asarray([], dtype=int)
                date_anchor = self.source_anchor_
            else:
                if len(allowed) < budget:
                    if not allow_partial_budget or len(allowed) == 0:
                        raise ValueError(
                            f"Date {dates[row]!r}, row {row} has only {len(allowed)} preceding anchors for B={budget}."
                        )
                chosen = allowed[: min(budget, len(allowed))]
                date_anchor = X[chosen[0]] if budget == 1 else np.nanmedian(X[chosen], axis=0)
                if not np.isfinite(date_anchor).all():
                    raise ValueError("A date-level blank anchor contains non-finite values.")
                # Blank adjustment is defined for physical sensor groups only;
                # ungrouped covariates such as temperature/humidity remain as
                # observed rather than being shifted by a sensor anchor.
                for columns in self.groups.values():
                    adjusted[row, columns] = X[row, columns] - (
                        date_anchor[columns] - self.source_anchor_[columns]
                    )
            receipt.append(
                AnchorUse(
                    row_index=int(row),
                    date=str(dates[row]),
                    budget=budget,
                    effective_budget=int(len(chosen)),
                    anchor_indices=tuple(int(value) for value in chosen.tolist()),
                    latest_anchor_order=None if not len(chosen) else float(np.max(order[chosen])),
                    scored_order=float(order[row]),
                )
            )
        return adjusted, tuple(receipt)
