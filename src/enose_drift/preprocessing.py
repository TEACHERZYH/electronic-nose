"""Training-only preprocessing with explicit fitted-state auditing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustTrainTransformer:
    """Median imputation, optional train-quantile clipping, and robust scaling."""

    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    minimum_scale: float = 1e-12

    def fit(self, X: np.ndarray) -> "RobustTrainTransformer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty matrix.")
        self.impute_ = np.nanmedian(X, axis=0)
        self.impute_[~np.isfinite(self.impute_)] = 0.0
        filled = np.where(np.isfinite(X), X, self.impute_)
        self.lower_ = np.quantile(filled, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(filled, self.upper_quantile, axis=0)
        clipped = np.clip(filled, self.lower_, self.upper_)
        self.center_ = np.median(clipped, axis=0)
        q25, q75 = np.quantile(clipped, [0.25, 0.75], axis=0)
        scale = q75 - q25
        fallback = np.std(clipped, axis=0)
        scale = np.where(scale > self.minimum_scale, scale, fallback)
        self.scale_ = np.where(scale > self.minimum_scale, scale, 1.0)
        self.n_features_in_ = X.shape[1]
        self.fit_row_count_ = X.shape[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "center_"):
            raise RuntimeError("Transformer is not fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from the fitted training matrix.")
        filled = np.where(np.isfinite(X), X, self.impute_)
        return (np.clip(filled, self.lower_, self.upper_) - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def audit_record(self) -> dict:
        if not hasattr(self, "center_"):
            raise RuntimeError("Transformer is not fitted.")
        return {
            "fit_row_count": int(self.fit_row_count_),
            "n_features": int(self.n_features_in_),
            "lower_quantile": float(self.lower_quantile),
            "upper_quantile": float(self.upper_quantile),
            "center_finite": bool(np.isfinite(self.center_).all()),
            "scale_positive": bool(np.all(self.scale_ > 0)),
        }


def select_group_columns(groups: dict, selected_groups: list | tuple) -> tuple[np.ndarray, dict]:
    """Return selected column indices and a compact remapped group dictionary."""

    columns: list[int] = []
    remapped: dict = {}
    cursor = 0
    for group in selected_groups:
        original = np.asarray(groups[group], dtype=int)
        columns.extend(original.tolist())
        remapped[group] = np.arange(cursor, cursor + len(original), dtype=int)
        cursor += len(original)
    return np.asarray(columns, dtype=int), remapped
