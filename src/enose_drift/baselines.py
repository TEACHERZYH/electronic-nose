"""Information-access-matched baseline models and sensor-group rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from .preprocessing import RobustTrainTransformer


_LOGISTIC_NAMES = {"multinomial", "logistic", "pooled_group_lasso", "tcss"}
_LOGISTIC_MAX_ITER_SCHEDULE = (3000, 10000)
HARD_PREDICTION_RULE = "predict_proba_argmax"


def probability_argmax_prediction(estimator: object, X: np.ndarray) -> np.ndarray:
    """Return the hard labels implied by the estimator's reported probabilities."""

    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Estimator probabilities do not match its class labels.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Estimator probabilities must be finite.")
    return classes[np.argmax(probabilities, axis=1)]


def _make_estimator(
    name: str, *, C: float = 1.0, seed: int = 11, max_iter: int = 3000
):
    normalized = name.lower().replace("-", "_")
    if normalized == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if normalized in {"rbf_svm", "mrmr_svm", "coral_svm"}:
        return SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    if normalized in _LOGISTIC_NAMES:
        return LogisticRegression(
            C=C, solver="lbfgs", max_iter=int(max_iter), random_state=seed
        )
    raise ValueError(f"Unknown model name: {name}")


def _fit_with_convergence_guard(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    seed: int,
    logistic_max_iter: Sequence[int] = _LOGISTIC_MAX_ITER_SCHEDULE,
) -> tuple[object, dict]:
    """Fit one candidate and record whether its optimizer converged.

    Logistic fits retain the same solver and objective while increasing only
    the iteration ceiling.  A candidate is eligible only after a warning-free
    fit.  Other estimator families do not expose the same convergence signal
    and are fitted once with their existing settings.
    """

    normalized = model_name.lower().replace("-", "_")
    if normalized not in _LOGISTIC_NAMES:
        estimator = _make_estimator(model_name, C=C, seed=seed)
        estimator.fit(X, y)
        return estimator, {
            "converged": True,
            "n_iter": None,
            "max_iter": None,
            "fit_attempts": 1,
            "convergence_warning": "",
        }

    limits = tuple(dict.fromkeys(int(value) for value in logistic_max_iter))
    if not limits or any(value <= 0 for value in limits):
        raise ValueError("logistic_max_iter must contain positive iteration limits.")
    if any(right <= left for left, right in zip(limits, limits[1:])):
        raise ValueError("logistic_max_iter must be strictly increasing.")

    messages: list[str] = []
    estimator = None
    n_iter = None
    for attempt, limit in enumerate(limits, start=1):
        estimator = _make_estimator(
            model_name, C=float(C), seed=int(seed), max_iter=int(limit)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X, y)
        convergence_messages = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        messages.extend(convergence_messages)
        raw_n_iter = getattr(estimator, "n_iter_", None)
        n_iter = int(np.max(raw_n_iter)) if raw_n_iter is not None else None
        if not convergence_messages:
            return estimator, {
                "converged": True,
                "n_iter": n_iter,
                "max_iter": int(limit),
                "fit_attempts": int(attempt),
                "convergence_warning": " | ".join(messages),
            }

    assert estimator is not None
    return estimator, {
        "converged": False,
        "n_iter": n_iter,
        "max_iter": int(limits[-1]),
        "fit_attempts": len(limits),
        "convergence_warning": " | ".join(messages),
    }


@dataclass
class FittedProbabilisticModel:
    transformer: RobustTrainTransformer
    estimator: object
    model_name: str
    selected_C: float
    fit_converged: bool = True
    fit_n_iter: int | None = None
    fit_max_iter: int | None = None
    fit_attempts: int = 1

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
        transformed = self.transformer.transform(X)
        return probability_argmax_prediction(self.estimator, transformed)


def tune_and_fit_probabilistic_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int = 11,
    candidate_C: Sequence[float] = (0.1, 1.0, 10.0),
    logistic_max_iter: Sequence[int] = _LOGISTIC_MAX_ITER_SCHEDULE,
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
        estimator, fit_status = _fit_with_convergence_guard(
            model_name,
            Xtr,
            y_train,
            C=float(C),
            seed=int(seed),
            logistic_max_iter=logistic_max_iter,
        )
        prediction = probability_argmax_prediction(estimator, Xva)
        score = f1_score(y_validation, prediction, average="macro", zero_division=0)
        records.append(
            {
                "model": model_name,
                "C": float(C),
                "validation_macro_f1": float(score),
                "hard_prediction_rule": HARD_PREDICTION_RULE,
                **fit_status,
                "eligible": bool(fit_status["converged"]),
                "selected": False,
            }
        )
        if not fit_status["converged"]:
            continue
        key = (score, -float(C))
        if best is None or key > best[0]:
            best = (key, float(C))
    if best is None:
        raise RuntimeError(
            f"No converged {model_name} candidate remained after iteration limits "
            f"{tuple(int(value) for value in logistic_max_iter)}."
        )
    selected_C = best[1]
    estimator, final_status = _fit_with_convergence_guard(
        model_name,
        Xtr,
        y_train,
        C=selected_C,
        seed=int(seed),
        logistic_max_iter=logistic_max_iter,
    )
    if not final_status["converged"]:
        raise RuntimeError(
            f"Selected {model_name} candidate C={selected_C:g} did not converge "
            f"after {final_status['fit_attempts']} fit attempts."
        )
    for record in records:
        if float(record["C"]) == selected_C:
            record.update(
                {
                    "selected": True,
                    "final_converged": True,
                    "final_n_iter": final_status["n_iter"],
                    "final_max_iter": final_status["max_iter"],
                    "final_fit_attempts": final_status["fit_attempts"],
                    "final_convergence_warning": final_status["convergence_warning"],
                }
            )
    return (
        FittedProbabilisticModel(
            transformer,
            estimator,
            model_name,
            selected_C,
            fit_converged=True,
            fit_n_iter=final_status["n_iter"],
            fit_max_iter=final_status["max_iter"],
            fit_attempts=final_status["fit_attempts"],
        ),
        records,
    )


def fit_probabilistic_model_with_fixed_c(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    seed: int = 11,
    logistic_max_iter: Sequence[int] = _LOGISTIC_MAX_ITER_SCHEDULE,
) -> FittedProbabilisticModel:
    """Fit on the supplied labeled history after C has been frozen elsewhere."""

    transformer = RobustTrainTransformer().fit(X)
    transformed = transformer.transform(X)
    estimator, fit_status = _fit_with_convergence_guard(
        model_name,
        transformed,
        y,
        C=float(C),
        seed=int(seed),
        logistic_max_iter=logistic_max_iter,
    )
    if not fit_status["converged"]:
        raise RuntimeError(
            f"Fixed-C {model_name} model C={float(C):g} did not converge after "
            f"{fit_status['fit_attempts']} fit attempts."
        )
    return FittedProbabilisticModel(
        transformer,
        estimator,
        model_name,
        float(C),
        fit_converged=True,
        fit_n_iter=fit_status["n_iter"],
        fit_max_iter=fit_status["max_iter"],
        fit_attempts=fit_status["fit_attempts"],
    )


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
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PSD matrix power requires a square matrix.")
    if not np.isfinite(matrix).all() or not np.isfinite(power):
        raise ValueError("PSD matrix power inputs must be finite.")
    if regularization <= 0.0 or not np.isfinite(regularization):
        raise ValueError("PSD matrix power regularization must be finite and positive.")

    symmetric = (matrix + matrix.T) / 2.0
    regularized = symmetric + regularization * np.eye(matrix.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    clipped = np.maximum(eigenvalues, float(regularization))
    powered = (eigenvectors * np.power(clipped, power)) @ eigenvectors.T
    return np.asarray((powered + powered.T) / 2.0, dtype=float)


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
