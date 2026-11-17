"""Electronic-nose numerical routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed


Array = np.ndarray


def _softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -745.0, 50.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _as_index_groups(groups: Mapping[Hashable, Sequence[int]], p: int) -> tuple[list[Hashable], list[Array]]:
    names: list[Hashable] = []
    indices: list[Array] = []
    seen: set[int] = set()
    for name, raw_idx in groups.items():
        idx = np.asarray(raw_idx, dtype=int)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"Group {name!r} must contain at least one one-dimensional index.")
        if np.any(idx < 0) or np.any(idx >= p):
            raise ValueError(f"Group {name!r} contains an out-of-range feature index.")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"Group {name!r} contains duplicate feature indices.")
        overlap = seen.intersection(idx.tolist())
        if overlap:
            raise ValueError(f"Feature indices occur in more than one group: {sorted(overlap)[:5]}")
        seen.update(idx.tolist())
        names.append(name)
        indices.append(idx)
    if not names:
        raise ValueError("At least one feature group is required.")
    return names, indices


def _tail_weights(losses: Array, alpha: float) -> Array:
    """Return exact empirical CVaR weights for equally weighted environments."""

    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("Environment losses must be a non-empty vector.")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    mass = (1.0 - alpha) * losses.size
    order = np.argsort(-losses, kind="mergesort")
    weights = np.zeros_like(losses)
    remaining = mass
    for idx in order:
        take = min(1.0, remaining)
        if take <= 0:
            break
        weights[idx] = take / mass
        remaining -= take
    return weights


def group_proximal(weights: Array, groups: Sequence[Array], threshold: float) -> Array:
    """Apply a non-overlapping Frobenius group-lasso proximal operator."""

    out = np.asarray(weights, dtype=float).copy()
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    for idx in groups:
        block = out[idx, :]
        norm = float(np.linalg.norm(block))
        if norm <= threshold:
            out[idx, :] = 0.0
        else:
            out[idx, :] *= 1.0 - threshold / norm
    return out


class GroupSparseCVaRMultinomial:
    """Multiclass linear classifier optimized with proximal gradient.

    `alpha=0` gives the pooled mean-environment objective and is used for the
    matched pooled group-lasso selector.
    """

    def __init__(
        self,
        groups: Mapping[Hashable, Sequence[int]],
        *,
        alpha: float = 0.75,
        group_lambda: float = 0.01,
        max_iter: int = 500,
        tol: float = 1e-6,
        stable_iterations: int = 5,
        initial_step_size: float = 1.0,
        backtracking_factor: float = 0.5,
        max_backtracking: int = 60,
    ) -> None:
        self.groups = groups
        self.alpha = float(alpha)
        self.group_lambda = float(group_lambda)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.stable_iterations = int(stable_iterations)
        self.initial_step_size = float(initial_step_size)
        self.backtracking_factor = float(backtracking_factor)
        self.max_backtracking = int(max_backtracking)

    def _validate_hyperparameters(self) -> None:
        if not 0 <= self.alpha < 1:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        if self.group_lambda < 0:
            raise ValueError("group_lambda must be non-negative.")
        if self.max_iter < 1 or self.stable_iterations < 1:
            raise ValueError("Iteration counts must be positive.")
        if self.tol <= 0 or self.initial_step_size <= 0:
            raise ValueError("Tolerance and initial step size must be positive.")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must lie in (0, 1).")

    def _loss_gradient(self, X: Array, y_idx: Array, env: Array, W: Array, b: Array) -> tuple[float, Array, Array, Array]:
        n_classes = W.shape[1]
        environment_ids = np.unique(env)
        losses = np.empty(environment_ids.size, dtype=float)
        grad_w = np.zeros_like(W)
        grad_b = np.zeros_like(b)
        env_grad_w: list[Array] = []
        env_grad_b: list[Array] = []
        for j, environment in enumerate(environment_ids):
            take = np.flatnonzero(env == environment)
            logits = X[take] @ W + b
            prob = _softmax(logits)
            losses[j] = -np.mean(np.log(np.clip(prob[np.arange(take.size), y_idx[take]], 1e-15, 1.0)))
            residual = prob
            residual[np.arange(take.size), y_idx[take]] -= 1.0
            env_grad_w.append(X[take].T @ residual / take.size)
            env_grad_b.append(np.mean(residual, axis=0))
        tail_weights = _tail_weights(losses, self.alpha)
        for weight, gw, gb in zip(tail_weights, env_grad_w, env_grad_b):
            grad_w += weight * gw
            grad_b += weight * gb
        return float(np.dot(tail_weights, losses)), grad_w, grad_b, losses

    def _objective(self, X: Array, y_idx: Array, env: Array, W: Array, b: Array) -> tuple[float, float, Array]:
        smooth, _, _, losses = self._loss_gradient(X, y_idx, env, W, b)
        penalty = self.group_lambda * sum(float(np.linalg.norm(W[idx, :])) for idx in self.group_indices_)
        return smooth + penalty, smooth, losses

    def fit(self, X: Array, y: Array, environments: Array, *, classes: Sequence | None = None) -> "GroupSparseCVaRMultinomial":
        self._validate_hyperparameters()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        environments = np.asarray(environments)
        if X.ndim != 2 or y.ndim != 1 or environments.ndim != 1:
            raise ValueError("X must be two-dimensional; y and environments must be one-dimensional.")
        if not (len(X) == len(y) == len(environments)) or len(X) == 0:
            raise ValueError("X, y and environments must have the same non-zero row count.")
        if not np.isfinite(X).all():
            raise ValueError("TCSS input must be finite after training-only preprocessing.")
        if classes is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.asarray(classes)
        class_to_index = {value: idx for idx, value in enumerate(self.classes_.tolist())}
        try:
            y_idx = np.asarray([class_to_index[value] for value in y.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Observed class {exc.args[0]!r} is absent from the supplied class list.") from exc
        if self.classes_.size < 2:
            raise ValueError("At least two classes are required.")
        self.group_names_, self.group_indices_ = _as_index_groups(self.groups, X.shape[1])

        W = np.zeros((X.shape[1], self.classes_.size), dtype=float)
        b = np.zeros(self.classes_.size, dtype=float)
        objective, smooth, losses = self._objective(X, y_idx, environments, W, b)
        self.objective_history_ = [objective]
        self.environment_loss_history_ = [losses.copy()]
        step = self.initial_step_size
        stable_count = 0
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            smooth, grad_w, grad_b, _ = self._loss_gradient(X, y_idx, environments, W, b)
            accepted = False
            local_step = step
            for _ in range(self.max_backtracking):
                candidate_w = group_proximal(W - local_step * grad_w, self.group_indices_, local_step * self.group_lambda)
                candidate_b = b - local_step * grad_b
                candidate_obj, candidate_smooth, candidate_losses = self._objective(
                    X, y_idx, environments, candidate_w, candidate_b
                )
                delta_w = candidate_w - W
                delta_b = candidate_b - b
                quadratic_upper = (
                    smooth
                    + float(np.sum(grad_w * delta_w) + np.dot(grad_b, delta_b))
                    + float(np.sum(delta_w * delta_w) + np.dot(delta_b, delta_b)) / (2.0 * local_step)
                )
                if candidate_smooth <= quadratic_upper + 1e-12 or candidate_obj < objective:
                    accepted = True
                    break
                local_step *= self.backtracking_factor
            if not accepted:
                raise RuntimeError("Proximal backtracking failed to find a valid descent step.")

            relative_change = abs(objective - candidate_obj) / max(1.0, abs(objective))
            W, b, objective = candidate_w, candidate_b, candidate_obj
            self.objective_history_.append(objective)
            self.environment_loss_history_.append(candidate_losses.copy())
            if relative_change < self.tol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= self.stable_iterations:
                self.converged_ = True
                self.n_iter_ = iteration
                break
            step = min(self.initial_step_size, local_step / self.backtracking_factor)
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = W
        self.intercept_ = b
        self.step_size_ = local_step
        self.environment_losses_ = self.environment_loss_history_[-1]
        return self

    def decision_function(self, X: Array) -> Array:
        if not hasattr(self, "coef_"):
            raise RuntimeError("The model must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: Array) -> Array:
        return _softmax(self.decision_function(X))

    def predict(self, X: Array) -> Array:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def group_magnitudes(self) -> dict[Hashable, float]:
        return {
            name: float(np.linalg.norm(self.coef_[idx, :]))
            for name, idx in zip(self.group_names_, self.group_indices_)
        }


@dataclass(frozen=True)
class StabilityRanking:
    group_names: tuple[Hashable, ...]
    frequency: Array
    mean_magnitude: Array
    ranking: tuple[Hashable, ...]
    replicate_rankings: tuple[tuple[Hashable, ...], ...]
    converged_fraction: float
    replicate_converged: tuple[bool, ...] = ()

    def top(self, k: int) -> tuple[Hashable, ...]:
        if not 1 <= k <= len(self.ranking):
            raise ValueError("k is outside the available group range.")
        return self.ranking[:k]

    def as_records(self) -> list[dict]:
        rank_position = {name: rank + 1 for rank, name in enumerate(self.ranking)}
        return [
            {
                "group": str(name),
                "rank": rank_position[name],
                "selection_frequency": float(self.frequency[i]),
                "mean_group_magnitude": float(self.mean_magnitude[i]),
            }
            for i, name in enumerate(self.group_names)
        ]


def make_consecutive_environments(dates: Array, n_environments: int = 6) -> Array:
    dates = np.asarray(dates)
    unique_dates = np.asarray(sorted(np.unique(dates).tolist()))
    if unique_dates.size < n_environments:
        raise ValueError("There are fewer unique dates than requested temporal environments.")
    chunks = np.array_split(unique_dates, n_environments)
    date_to_environment = {date: env_id for env_id, chunk in enumerate(chunks) for date in chunk.tolist()}
    return np.asarray([date_to_environment[date] for date in dates.tolist()], dtype=int)


def _rank(names: Sequence[Hashable], frequency: Array, magnitude: Array) -> tuple[Hashable, ...]:
    order = sorted(
        range(len(names)),
        key=lambda idx: (-float(frequency[idx]), -float(magnitude[idx]), str(names[idx])),
    )
    return tuple(names[idx] for idx in order)


def _cluster_resample_indices(dates: Array, environments: Array, rng: np.random.Generator) -> tuple[Array, Array]:
    selected_rows: list[Array] = []
    selected_environments: list[Array] = []
    for environment in np.unique(environments):
        env_dates = np.unique(dates[environments == environment])
        sampled_dates = rng.choice(env_dates, size=len(env_dates), replace=True)
        for sampled_date in sampled_dates:
            rows = np.flatnonzero((dates == sampled_date) & (environments == environment))
            selected_rows.append(rows)
            selected_environments.append(np.full(rows.size, environment, dtype=int))
    return np.concatenate(selected_rows), np.concatenate(selected_environments)


def _fit_one_cluster_resample(
    replicate_seed: int,
    X: Array,
    y: Array,
    dates: Array,
    environments: Array,
    groups: Mapping[Hashable, Sequence[int]],
    classes: Array,
    model_kwargs: dict,
    coefficient_tolerance: float,
) -> tuple[Array, Array, tuple[Hashable, ...], bool]:
    rng = np.random.default_rng(replicate_seed)
    rows, sampled_env = _cluster_resample_indices(dates, environments, rng)
    model = GroupSparseCVaRMultinomial(groups, **model_kwargs).fit(
        X[rows], y[rows], sampled_env, classes=classes
    )
    magnitudes_dict = model.group_magnitudes()
    names = tuple(groups.keys())
    magnitude = np.asarray([magnitudes_dict[name] for name in names])
    inclusion = magnitude > coefficient_tolerance
    ranking = _rank(names, inclusion.astype(float), magnitude)
    return inclusion, magnitude, ranking, model.converged_


def clustered_stability_ranking(
    X: Array,
    y: Array,
    dates: Array,
    groups: Mapping[Hashable, Sequence[int]],
    *,
    n_environments: int = 6,
    n_resamples: int = 200,
    seed: int = 11,
    coefficient_tolerance: float = 1e-6,
    n_jobs: int = 1,
    model_kwargs: dict | None = None,
) -> StabilityRanking:
    X, y, dates = np.asarray(X, float), np.asarray(y), np.asarray(dates)
    if not (len(X) == len(y) == len(dates)):
        raise ValueError("X, y and dates must have the same row count.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    environments = make_consecutive_environments(dates, n_environments)
    classes = np.unique(y)
    names = tuple(groups.keys())
    kwargs = dict(model_kwargs or {})
    seed_sequence = np.random.SeedSequence(seed).spawn(n_resamples)
    replicate_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in seed_sequence]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_one_cluster_resample)(
            replicate_seed, X, y, dates, environments, groups, classes, kwargs, coefficient_tolerance
        )
        for replicate_seed in replicate_seeds
    )
    inclusion = np.stack([result[0] for result in results])
    magnitude = np.stack([result[1] for result in results])
    frequency = np.mean(inclusion, axis=0)
    mean_magnitude = np.mean(magnitude, axis=0)
    ranking = _rank(names, frequency, mean_magnitude)
    return StabilityRanking(
        group_names=names,
        frequency=frequency,
        mean_magnitude=mean_magnitude,
        ranking=ranking,
        replicate_rankings=tuple(result[2] for result in results),
        converged_fraction=float(np.mean([result[3] for result in results])),
        replicate_converged=tuple(bool(result[3]) for result in results),
    )


def stratified_row_stability_ranking(
    X: Array,
    y: Array,
    groups: Mapping[Hashable, Sequence[int]],
    *,
    n_resamples: int = 200,
    seed: int = 11,
    coefficient_tolerance: float = 1e-6,
    n_jobs: int = 1,
    model_kwargs: dict | None = None,
) -> StabilityRanking:
    """UCI Batch-1 fallback; it does not test temporal stability."""

    X, y = np.asarray(X, float), np.asarray(y)
    classes = np.unique(y)
    names = tuple(groups.keys())
    kwargs = dict(model_kwargs or {})
    kwargs["alpha"] = 0.0
    seed_sequence = np.random.SeedSequence(seed).spawn(n_resamples)

    def fit_one(child_seed: np.random.SeedSequence):
        rng = np.random.default_rng(child_seed)
        rows = np.concatenate(
            [rng.choice(np.flatnonzero(y == label), size=np.sum(y == label), replace=True) for label in classes]
        )
        env = np.zeros(rows.size, dtype=int)
        model = GroupSparseCVaRMultinomial(groups, **kwargs).fit(X[rows], y[rows], env, classes=classes)
        magnitudes_dict = model.group_magnitudes()
        magnitude = np.asarray([magnitudes_dict[name] for name in names])
        inclusion = magnitude > coefficient_tolerance
        return inclusion, magnitude, _rank(names, inclusion.astype(float), magnitude), model.converged_

    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(fit_one)(child) for child in seed_sequence)
    inclusion = np.stack([result[0] for result in results])
    magnitude = np.stack([result[1] for result in results])
    frequency = np.mean(inclusion, axis=0)
    mean_magnitude = np.mean(magnitude, axis=0)
    return StabilityRanking(
        group_names=names,
        frequency=frequency,
        mean_magnitude=mean_magnitude,
        ranking=_rank(names, frequency, mean_magnitude),
        replicate_rankings=tuple(result[2] for result in results),
        converged_fraction=float(np.mean([result[3] for result in results])),
        replicate_converged=tuple(bool(result[3]) for result in results),
    )


def single_fit_group_ranking(model: GroupSparseCVaRMultinomial) -> tuple[Hashable, ...]:
    magnitudes = model.group_magnitudes()
    names = tuple(magnitudes.keys())
    values = np.asarray([magnitudes[name] for name in names])
    return _rank(names, (values > 0).astype(float), values)
