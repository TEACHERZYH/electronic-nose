"""Information-access-matched baseline models and sensor-group rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from .preprocessing import RobustTrainTransformer


def _make_estimator(name: str, *, C: float = 1.0, seed: int = 11):
    normalized = name.lower().replace("-", "_")
    if normalized == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if normalized in {"rbf_svm", "mrmr_svm", "coral_svm"}:
        return SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    if normalized in {"multinomial", "logistic", "pooled_group_lasso", "tcss"}:
        return LogisticRegression(C=C, solver="lbfgs", max_iter=3000, random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


@dataclass
class FittedProbabilisticModel:
    transformer: RobustTrainTransformer
    estimator: object
    model_name: str
    selected_C: float

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self.transformer.transform(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        transformed = self.transformer.transform(X)
        if hasattr(self.estimator, "decision_function"):
            logits = self.estimator.decision_function(transformed)
            if logits.ndim == 1:
                logits = np.column_stack([-logits, logits])
            return logits
        probability = np.clip(self.estimator.predict_proba(transformed), 1e-15, 1.0)
        return np.log(probability)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def tune_and_fit_probabilistic_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int = 11,
    candidate_C: Sequence[float] = (0.1, 1.0, 10.0),
) -> tuple[FittedProbabilisticModel, list[dict]]:
    """Select hyperparameters on validation rows and refit on training only.

    The final returned model deliberately remains fitted on training rows.  A
    caller may explicitly refit on train+validation only after freezing C and
    recording that protocol choice.
    """

    transformer = RobustTrainTransformer().fit(X_train)
    Xtr = transformer.transform(X_train)
    Xva = transformer.transform(X_validation)
    grid = (1.0,) if model_name.lower() == "lda" else tuple(candidate_C)
    records: list[dict] = []
    best = None
    for C in grid:
        estimator = _make_estimator(model_name, C=C, seed=seed)
        estimator.fit(Xtr, y_train)
        prediction = estimator.predict(Xva)
        score = f1_score(y_validation, prediction, average="macro", zero_division=0)
        records.append({"model": model_name, "C": float(C), "validation_macro_f1": float(score)})
        key = (score, -float(C))
        if best is None or key > best[0]:
            best = (key, float(C))
    assert best is not None
    selected_C = best[1]
    estimator = _make_estimator(model_name, C=selected_C, seed=seed).fit(Xtr, y_train)
    return FittedProbabilisticModel(transformer, estimator, model_name, selected_C), records


def fit_probabilistic_model_with_fixed_c(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    seed: int = 11,
) -> FittedProbabilisticModel:
    """Fit on the supplied labeled history after C has been frozen elsewhere."""

    transformer = RobustTrainTransformer().fit(X)
    transformed = transformer.transform(X)
    estimator = _make_estimator(model_name, C=float(C), seed=int(seed)).fit(transformed, y)
    return FittedProbabilisticModel(transformer, estimator, model_name, float(C))


def mrmr_group_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    *,
    seed: int = 11,
) -> tuple[Hashable, ...]:
    """Greedy group mRMR ranking using training rows only."""

    X = RobustTrainTransformer().fit_transform(X_train)
    relevance_feature = mutual_info_classif(X, y_train, random_state=seed)
    names = list(groups.keys())
    relevance = {
        name: float(np.mean(relevance_feature[np.asarray(groups[name], dtype=int)])) for name in names
    }
    selected: list[Hashable] = []
    remaining = set(names)
    while remaining:
        best_name = None
        best_key = None
        for name in remaining:
            if not selected:
                score = relevance[name]
            else:
                current = X[:, np.asarray(groups[name], dtype=int)]
                redundancy_values: list[float] = []
                for chosen in selected:
                    other = X[:, np.asarray(groups[chosen], dtype=int)]
                    corr = np.corrcoef(current.T, other.T)
                    block = corr[: current.shape[1], current.shape[1] :]
                    finite = np.abs(block[np.isfinite(block)])
                    redundancy_values.append(float(np.mean(finite)) if finite.size else 0.0)
                score = relevance[name] - float(np.mean(redundancy_values))
            key = (score, relevance[name], str(name))
            if best_key is None or key > best_key:
                best_key, best_name = key, name
        selected.append(best_name)
        remaining.remove(best_name)
    return tuple(selected)


def _symmetric_psd_power(matrix: np.ndarray, power: float, regularization: float) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    regularized = matrix + regularization * np.eye(matrix.shape[0])
    powered = fractional_matrix_power(regularized, power)
    return np.asarray(np.real_if_close(powered), dtype=float)


@dataclass
class CORALTransform:
    source_mean: np.ndarray
    target_mean: np.ndarray
    mapping: np.ndarray

    def transform_source(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.source_mean) @ self.mapping + self.target_mean

    def transform_target(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float)


def fit_coral_transform(X_source: np.ndarray, X_target_unlabeled: np.ndarray, regularization: float = 1e-5) -> CORALTransform:
    """Fit CORAL using source rows and the explicitly allowed unlabeled target."""

    Xs = np.asarray(X_source, dtype=float)
    Xt = np.asarray(X_target_unlabeled, dtype=float)
    if Xs.ndim != 2 or Xt.ndim != 2 or Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Source and target must be matrices with matching feature dimensions.")
    source_mean = np.mean(Xs, axis=0)
    target_mean = np.mean(Xt, axis=0)
    source_cov = np.cov(Xs - source_mean, rowvar=False)
    target_cov = np.cov(Xt - target_mean, rowvar=False)
    mapping = _symmetric_psd_power(source_cov, -0.5, regularization) @ _symmetric_psd_power(
        target_cov, 0.5, regularization
    )
    return CORALTransform(source_mean, target_mean, mapping)


def fit_coral_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_target_unlabeled: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 11,
) -> tuple[CORALTransform, RobustTrainTransformer, SVC]:
    coral = fit_coral_transform(X_train, X_target_unlabeled)
    aligned_source = coral.transform_source(X_train)
    transformer = RobustTrainTransformer().fit(aligned_source)
    estimator = _make_estimator("coral_svm", C=C, seed=seed)
    estimator.fit(transformer.transform(aligned_source), y_train)
    return coral, transformer, estimator
