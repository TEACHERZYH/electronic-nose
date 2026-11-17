"""Formal primary-dataset selector, resource-grid, and mask-stress models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from .baselines import (
    HARD_PREDICTION_RULE,
    fit_coral_svm,
    fit_probabilistic_model_with_fixed_c,
    mrmr_group_ranking,
    probability_argmax_prediction,
    tune_and_fit_probabilistic_model,
)
from .calibration import TemperatureScaler
from .evaluation import PredictionBundle, bundle_from_model
from .masking import (
    apply_group_availability,
    availability_from_record,
    expand_structured_dropout_training,
    generate_evaluation_masks,
    group_centers,
)
from .preprocessing import RobustTrainTransformer, select_group_columns
from .protocol import primary_expanding_window_splits
from .tcss import (
    GroupSparseCVaRMultinomial,
    StabilityRanking,
    clustered_stability_ranking,
    make_consecutive_environments,
    single_fit_group_ranking,
)


@dataclass(frozen=True)
class SelectorRankings:
    tcss: tuple[Hashable, ...]
    single_fit: tuple[Hashable, ...]
    pooled_group_lasso: tuple[Hashable, ...]
    dense_cvar: tuple[Hashable, ...]
    mrmr: tuple[Hashable, ...]
    stability: StabilityRanking
    pooled_magnitudes: dict[Hashable, float]

    def for_method(self, method: str) -> tuple[Hashable, ...]:
        key = method.lower().replace("-", "_")
        if key == "tcss":
            return self.tcss
        if key in {"tcss_single_fit", "single_fit"}:
            return self.single_fit
        if key == "pooled_group_lasso":
            return self.pooled_group_lasso
        if key in {"dense_cvar", "no_group_sparsity"}:
            return self.dense_cvar
        if key in {"mrmr", "mrmr_svm"}:
            return self.mrmr
        raise ValueError(f"No sensor ranking is defined for method {method!r}.")


def _selector_kwargs(
    config: Mapping[str, Any],
    *,
    alpha: float | None = None,
    group_lambda: float | None = None,
) -> dict[str, Any]:
    settings = config["tcss"]
    return {
        "alpha": float(settings["alpha"] if alpha is None else alpha),
        "group_lambda": float(settings["group_lambda"] if group_lambda is None else group_lambda),
        "max_iter": int(settings["max_iterations"]),
        "tol": float(settings["relative_objective_tolerance"]),
        "stable_iterations": int(settings["stable_iterations"]),
        "initial_step_size": float(settings.get("initial_step_size", 1.0)),
        "backtracking_factor": float(settings.get("backtracking_factor", 0.5)),
    }


def fit_selector_rankings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_dates: np.ndarray,
    groups: Mapping[Hashable, Sequence[int]],
    config: Mapping[str, Any],
    *,
    n_jobs: int = 1,
) -> tuple[SelectorRankings, RobustTrainTransformer]:
    """Fit all three training-only group rankings used in matched comparisons."""

    transformer = RobustTrainTransformer().fit(X_train)
    standardized = transformer.transform(X_train)
    settings = config["tcss"]
    stability = clustered_stability_ranking(
        standardized,
        y_train,
        train_dates,
        groups,
        n_environments=int(settings["environments"]),
        n_resamples=int(settings["stability_resamples"]),
        seed=int(settings["seeds"][0]),
        coefficient_tolerance=float(settings["coefficient_tolerance"]),
        n_jobs=n_jobs,
        model_kwargs=_selector_kwargs(config),
    )
    environments = make_consecutive_environments(train_dates, int(settings["environments"]))
    single_model = GroupSparseCVaRMultinomial(groups, **_selector_kwargs(config)).fit(
        standardized, y_train, environments
    )
    pooled = GroupSparseCVaRMultinomial(groups, **_selector_kwargs(config, alpha=0.0)).fit(
        standardized, y_train, environments
    )
    dense_cvar = GroupSparseCVaRMultinomial(
        groups, **_selector_kwargs(config, group_lambda=0.0)
    ).fit(standardized, y_train, environments)
    pooled_ranking = single_fit_group_ranking(pooled)
    mrmr_ranking = mrmr_group_ranking(X_train, y_train, groups, seed=int(settings["seeds"][0]))
    return (
        SelectorRankings(
            tcss=stability.ranking,
            single_fit=single_fit_group_ranking(single_model),
            pooled_group_lasso=pooled_ranking,
            dense_cvar=single_fit_group_ranking(dense_cvar),
            mrmr=mrmr_ranking,
            stability=stability,
            pooled_magnitudes=pooled.group_magnitudes(),
        ),
        transformer,
    )


def _split_indices(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "split" not in manifest or "label" not in manifest:
        raise ValueError("Scored manifest must contain split and label columns.")
    return tuple(np.flatnonzero(manifest["split"].eq(name).to_numpy()) for name in ("train", "validation", "test"))


def _filter_method_specs(
    available: Sequence[tuple[str, str]], requested: Sequence[str] | None
) -> tuple[tuple[str, str], ...]:
    if requested is None:
        return tuple(available)
    requested_keys = tuple(str(name).strip().lower().replace("-", "_") for name in requested)
    if not requested_keys:
        raise ValueError("methods must contain at least one method when supplied.")
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("methods contains duplicate method names.")
    available_by_key = {
        method.lower().replace("-", "_"): (method, model_name)
        for method, model_name in available
    }
    unknown = sorted(set(requested_keys).difference(available_by_key))
    if unknown:
        raise ValueError(f"Unknown requested methods: {unknown}")
    return tuple(available_by_key[key] for key in requested_keys)


def run_matched_resource_grid(
    X_by_budget: Mapping[int, np.ndarray],
    scored_manifest: pd.DataFrame,
    groups: Mapping[Hashable, Sequence[int]],
    rankings: SelectorRankings,
    config: Mapping[str, Any],
    *,
    run_id: str,
    dataset: str = "primary",
    always_include_columns: Sequence[int] = (),
    methods: Sequence[str] | None = None,
) -> tuple[list[PredictionBundle], list[dict[str, Any]]]:
    """Run TCSS, pooled group lasso, and mRMR under common K/B access."""

    train, validation, test = _split_indices(scored_manifest)
    y = scored_manifest["label"].to_numpy()
    sample_ids = scored_manifest["sample_id"].to_numpy()
    dates = scored_manifest["date"].to_numpy()
    bundles: list[PredictionBundle] = []
    tuning_records: list[dict[str, Any]] = []
    seeds = [int(value) for value in config["tcss"]["seeds"]]
    sensor_budgets = [int(value) for value in config["tcss"]["sensor_budgets"]]
    anchor_budgets = [int(value) for value in config["anchors"]["budgets"]]
    method_specs = _filter_method_specs(
        (
            ("TCSS", "multinomial"),
            ("pooled_group_lasso", "multinomial"),
            ("mRMR-SVM", "rbf_svm"),
        ),
        methods,
    )
    always_columns = tuple(int(value) for value in always_include_columns)

    for blank_budget in anchor_budgets:
        X = np.asarray(X_by_budget[blank_budget], dtype=float)
        if len(X) != len(scored_manifest):
            raise ValueError(f"B={blank_budget} matrix does not align with the common scoring manifest.")
        for k in sensor_budgets:
            for method, model_name in method_specs:
                ranking = rankings.for_method(method)
                selected = ranking[:k]
                columns, _ = select_group_columns(dict(groups), selected)
                if always_columns:
                    columns = np.concatenate([columns, np.asarray(always_columns, dtype=int)])
                for seed in seeds:
                    model, tuning = tune_and_fit_probabilistic_model(
                        model_name,
                        X[np.ix_(train, columns)],
                        y[train],
                        X[np.ix_(validation, columns)],
                        y[validation],
                        seed=seed,
                    )
                    common_metadata = {
                        "sensor_budget": k,
                        "blank_budget": blank_budget,
                        "selected_groups": [str(value) for value in selected],
                        "always_include_columns": list(always_columns),
                        "information_access": "blank_matched",
                    }
                    validation_bundle = bundle_from_model(
                        model,
                        X[np.ix_(validation, columns)],
                        y[validation],
                        run_id=run_id,
                        dataset=dataset,
                        method=method,
                        split="validation",
                        seed=seed,
                        sample_ids=sample_ids[validation],
                        date_ids=dates[validation],
                        metadata=common_metadata,
                    )
                    test_bundle = bundle_from_model(
                        model,
                        X[np.ix_(test, columns)],
                        y[test],
                        run_id=run_id,
                        dataset=dataset,
                        method=method,
                        split="test",
                        seed=seed,
                        sample_ids=sample_ids[test],
                        date_ids=dates[test],
                        metadata=common_metadata,
                    )
                    scaler = TemperatureScaler().fit(
                        validation_bundle.logits,
                        validation_bundle.y_true,
                        validation_bundle.classes,
                    )
                    calibrated_test = PredictionBundle(
                        run_id=test_bundle.run_id,
                        dataset=test_bundle.dataset,
                        method=test_bundle.method,
                        split=test_bundle.split,
                        seed=test_bundle.seed,
                        sample_ids=test_bundle.sample_ids,
                        date_ids=test_bundle.date_ids,
                        y_true=test_bundle.y_true,
                        classes=test_bundle.classes,
                        probabilities=scaler.predict_proba(test_bundle.logits),
                        logits=test_bundle.logits,
                        metadata={
                            **(test_bundle.metadata or {}),
                            "calibration": scaler.audit_record(),
                        },
                    )
                    calibrated_test.validate()
                    unscaled_test = PredictionBundle(
                        run_id=test_bundle.run_id,
                        dataset=test_bundle.dataset,
                        method=f"{test_bundle.method}-unscaled",
                        split=test_bundle.split,
                        seed=test_bundle.seed,
                        sample_ids=test_bundle.sample_ids,
                        date_ids=test_bundle.date_ids,
                        y_true=test_bundle.y_true,
                        classes=test_bundle.classes,
                        probabilities=test_bundle.probabilities,
                        logits=test_bundle.logits,
                        metadata={**(test_bundle.metadata or {}), "calibration": "none"},
                    )
                    bundles.extend([validation_bundle, calibrated_test, unscaled_test])
                    tuning_records.extend(
                        {
                            **record,
                            "method": method,
                            "seed": seed,
                            "sensor_budget": k,
                            "blank_budget": blank_budget,
                        }
                        for record in tuning
                    )
    return bundles, tuning_records


def run_source_only_baselines(
    X: np.ndarray,
    scored_manifest: pd.DataFrame,
    *,
    run_id: str,
    seeds: Sequence[int],
    sensor_count: int | None = None,
    methods: Sequence[str] | None = None,
) -> tuple[list[PredictionBundle], list[dict[str, Any]]]:
    train, validation, test = _split_indices(scored_manifest)
    y = scored_manifest["label"].to_numpy()
    sample_ids = scored_manifest["sample_id"].to_numpy()
    dates = scored_manifest["date"].to_numpy()
    bundles: list[PredictionBundle] = []
    tuning_records: list[dict[str, Any]] = []
    method_specs = _filter_method_specs(
        (("LDA", "lda"), ("RBF-SVM", "rbf_svm")), methods
    )
    for method, model_name in method_specs:
        for seed in map(int, seeds):
            model, tuning = tune_and_fit_probabilistic_model(
                model_name, X[train], y[train], X[validation], y[validation], seed=seed
            )
            metadata = {
                "information_access": "source_only",
                "sensor_budget": int(sensor_count) if sensor_count is not None else None,
                "predictor_count": int(X.shape[1]),
            }
            bundles.append(
                bundle_from_model(
                    model,
                    X[test],
                    y[test],
                    run_id=run_id,
                    dataset="primary",
                    method=method,
                    split="test",
                    seed=seed,
                    sample_ids=sample_ids[test],
                    date_ids=dates[test],
                    metadata=metadata,
                )
            )
            tuning_records.extend({**record, "method": method, "seed": seed} for record in tuning)
    return bundles, tuning_records


def run_coral_svm_baseline(
    X: np.ndarray,
    scored_manifest: pd.DataFrame,
    *,
    run_id: str,
    seeds: Sequence[int],
    sensor_count: int,
    candidate_c: Sequence[float] = (0.1, 1.0, 10.0),
) -> tuple[list[PredictionBundle], list[dict[str, Any]]]:
    """Run a transductive CORAL-SVM with explicitly unlabeled target access."""

    train, validation, test = _split_indices(scored_manifest)
    y = scored_manifest["label"].to_numpy()
    sample_ids = scored_manifest["sample_id"].to_numpy()
    dates = scored_manifest["date"].to_numpy()
    bundles: list[PredictionBundle] = []
    records: list[dict[str, Any]] = []
    for seed in map(int, seeds):
        best: tuple[tuple[float, float], float] | None = None
        for candidate in map(float, candidate_c):
            coral, transformer, estimator = fit_coral_svm(
                X[train], y[train], X[validation], C=candidate, seed=seed
            )
            validation_features = transformer.transform(coral.transform_target(X[validation]))
            prediction = probability_argmax_prediction(estimator, validation_features)
            labels = np.unique(y)
            recalls = []
            f1_values = []
            for label in labels:
                true_positive = np.sum((y[validation] == label) & (prediction == label))
                false_positive = np.sum((y[validation] != label) & (prediction == label))
                false_negative = np.sum((y[validation] == label) & (prediction != label))
                precision = true_positive / max(1, true_positive + false_positive)
                recall = true_positive / max(1, true_positive + false_negative)
                recalls.append(recall)
                f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
            score = float(np.mean(f1_values))
            records.append(
                {
                    "method": "CORAL-SVM",
                    "seed": seed,
                    "C": candidate,
                    "validation_macro_f1": score,
                    "hard_prediction_rule": HARD_PREDICTION_RULE,
                    "target_access_for_tuning": "validation_unlabeled_covariance_and_validation_labels_for_C",
                }
            )
            key = (score, -candidate)
            if best is None or key > best[0]:
                best = (key, candidate)
        assert best is not None
        selected_c = best[1]
        coral, transformer, estimator = fit_coral_svm(
            X[train], y[train], X[test], C=selected_c, seed=seed
        )
        test_features = transformer.transform(coral.transform_target(X[test]))
        probabilities = estimator.predict_proba(test_features)
        bundle = PredictionBundle(
            run_id=run_id,
            dataset="primary",
            method="CORAL-SVM",
            split="test",
            seed=seed,
            sample_ids=sample_ids[test],
            date_ids=dates[test],
            y_true=y[test],
            classes=np.asarray(estimator.classes_),
            probabilities=np.asarray(probabilities, dtype=float),
            logits=np.log(np.clip(probabilities, 1e-15, 1.0)),
            metadata={
                "information_access": "unlabeled_locked_test_covariance",
                "sensor_budget": int(sensor_count),
                "predictor_count": int(X.shape[1]),
                "selected_C": float(selected_c),
                "hard_prediction_rule": HARD_PREDICTION_RULE,
            },
        )
        bundle.validate()
        bundles.append(bundle)
    return bundles, records


def run_primary_expanding_window(
    X: np.ndarray,
    scored_manifest: pd.DataFrame,
    groups: Mapping[Hashable, Sequence[int]],
    config: Mapping[str, Any],
    *,
    run_id: str,
    always_include_columns: Sequence[int] = (),
    n_jobs: int = 1,
) -> tuple[list[PredictionBundle], list[dict[str, Any]], pd.DataFrame]:
    """Run the frozen last-six-date supplemental forward origins."""

    n_folds = int(config["supplement"]["primary_expanding_window_min_origins"])
    unique_dates = scored_manifest["date"].drop_duplicates().tolist()
    folds = primary_expanding_window_splits(
        scored_manifest,
        first_test_index=len(unique_dates) - n_folds,
        n_folds=n_folds,
    )
    y_all = scored_manifest["label"].to_numpy()
    sample_ids = scored_manifest["sample_id"].to_numpy()
    dates_all = scored_manifest["date"].to_numpy()
    bundles: list[PredictionBundle] = []
    tuning_records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    always = tuple(int(value) for value in always_include_columns)
    k = int(config["primary_comparison"]["sensors"])

    for origin_index, fold in enumerate(folds, start=1):
        train = fold.train.index.to_numpy(dtype=int)
        validation = fold.validation.index.to_numpy(dtype=int)
        history = fold.history.index.to_numpy(dtype=int)
        test = fold.test.index.to_numpy(dtype=int)
        rankings, _ = fit_selector_rankings(
            X[train],
            y_all[train],
            dates_all[train],
            groups,
            config,
            n_jobs=n_jobs,
        )
        for method, ranking in (
            ("TCSS", rankings.tcss),
            ("pooled_group_lasso", rankings.pooled_group_lasso),
        ):
            selected = ranking[:k]
            columns, _ = select_group_columns(dict(groups), selected)
            if always:
                columns = np.concatenate([columns, np.asarray(always, dtype=int)])
            for rank, group in enumerate(ranking, start=1):
                ranking_records.append(
                    {
                        "origin_index": origin_index,
                        "test_date": str(fold.test_key),
                        "method": method,
                        "rank": rank,
                        "group": str(group),
                        "selected_for_k": bool(rank <= k),
                        "selection_fit_dates": ";".join(map(str, fold.train_keys)),
                    }
                )
            for seed in map(int, config["tcss"]["seeds"]):
                tuned, tuning = tune_and_fit_probabilistic_model(
                    "multinomial",
                    X[np.ix_(train, columns)],
                    y_all[train],
                    X[np.ix_(validation, columns)],
                    y_all[validation],
                    seed=seed,
                )
                model = fit_probabilistic_model_with_fixed_c(
                    "multinomial",
                    X[np.ix_(history, columns)],
                    y_all[history],
                    C=tuned.selected_C,
                    seed=seed,
                )
                bundle = bundle_from_model(
                    model,
                    X[np.ix_(test, columns)],
                    y_all[test],
                    run_id=run_id,
                    dataset="primary",
                    method=method,
                    split="test",
                    seed=seed,
                    sample_ids=sample_ids[test],
                    date_ids=dates_all[test],
                    metadata={
                        "protocol": "primary_expanding_window",
                        "origin_index": origin_index,
                        "test_date": str(fold.test_key),
                        "history_dates": ";".join(map(str, fold.history_keys)),
                        "selection_fit_dates": ";".join(map(str, fold.train_keys)),
                        "validation_dates": ";".join(map(str, fold.validation_keys)),
                        "refit_on_full_history_after_tuning": True,
                        "future_date_used_for_fit": False,
                        "sensor_budget": k,
                        "blank_budget": 1,
                    },
                )
                bundles.append(bundle)
                tuning_records.extend(
                    {
                        **record,
                        "protocol": "primary_expanding_window",
                        "origin_index": origin_index,
                        "test_date": str(fold.test_key),
                        "method": method,
                        "seed": seed,
                    }
                    for record in tuning
                )
    return bundles, tuning_records, pd.DataFrame(ranking_records)


def _fit_mask_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    compact_groups: Mapping[Hashable, Sequence[int]],
    *,
    seed: int,
    max_fraction: float,
):
    expanded_X, expanded_y, _, centers = expand_structured_dropout_training(
        X_train,
        y_train,
        compact_groups,
        seed=seed,
        max_fraction=max_fraction,
        append_mask=True,
    )
    full_validation = apply_group_availability(
        X_validation,
        compact_groups,
        np.ones(len(compact_groups), dtype=bool),
        centers,
        append_mask=True,
    )
    model, tuning = tune_and_fit_probabilistic_model(
        "multinomial", expanded_X, expanded_y, full_validation, y_validation, seed=seed
    )
    return model, centers, full_validation, tuning


def run_mask_stress(
    X: np.ndarray,
    scored_manifest: pd.DataFrame,
    groups: Mapping[Hashable, Sequence[int]],
    influence_ranking: Sequence[Hashable],
    config: Mapping[str, Any],
    *,
    run_id: str,
    selected_k: int = 62,
    always_include_columns: Sequence[int] = (),
) -> tuple[list[PredictionBundle], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit mask-aware and unconditioned models, then apply identical masks."""

    train, validation, test = _split_indices(scored_manifest)
    y = scored_manifest["label"].to_numpy()
    sample_ids = scored_manifest["sample_id"].to_numpy()
    dates = scored_manifest["date"].to_numpy()
    selected = tuple(influence_ranking[:selected_k])
    columns, compact_groups = select_group_columns(dict(groups), selected)
    always_columns = tuple(int(value) for value in always_include_columns)
    if always_columns:
        columns = np.concatenate([columns, np.asarray(always_columns, dtype=int)])
    selected_X = np.asarray(X[:, columns], dtype=float)
    train_transformer = RobustTrainTransformer().fit(selected_X[train])
    standardized = train_transformer.transform(selected_X)
    settings = config["masking"]
    bundles: list[PredictionBundle] = []
    mask_receipts: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []

    for seed in map(int, config["tcss"]["seeds"]):
        mask_model, centers, full_validation, tuning = _fit_mask_model(
            standardized[train],
            y[train],
            standardized[validation],
            y[validation],
            compact_groups,
            seed=seed,
            max_fraction=float(settings["train_max_fraction"]),
        )
        unconditioned, unconditioned_tuning = tune_and_fit_probabilistic_model(
            "multinomial",
            standardized[train],
            y[train],
            standardized[validation],
            y[validation],
            seed=seed,
        )
        validation_mask_bundle = bundle_from_model(
            mask_model,
            full_validation,
            y[validation],
            run_id=run_id,
            dataset="primary",
            method="TCSS-mask-aware",
            split="validation",
            seed=seed,
            sample_ids=sample_ids[validation],
            date_ids=dates[validation],
        )
        mask_scaler = TemperatureScaler().fit(
            validation_mask_bundle.logits, validation_mask_bundle.y_true, validation_mask_bundle.classes
        )
        unconditioned_validation = bundle_from_model(
            unconditioned,
            standardized[validation],
            y[validation],
            run_id=run_id,
            dataset="primary",
            method="unconditioned",
            split="validation",
            seed=seed,
            sample_ids=sample_ids[validation],
            date_ids=dates[validation],
        )
        unconditioned_scaler = TemperatureScaler().fit(
            unconditioned_validation.logits,
            unconditioned_validation.y_true,
            unconditioned_validation.classes,
        )
        evaluation_centers = group_centers(standardized[train], compact_groups)
        for strategy in settings["strategies"]:
            records = generate_evaluation_masks(
                selected,
                settings["unavailable_counts"],
                [seed],
                strategy=strategy,
                influence_ranking=selected if strategy == "influence" else None,
            )
            for record in records:
                available = availability_from_record(selected, record)
                mask_features = apply_group_availability(
                    standardized[test], compact_groups, available, centers, append_mask=True
                )
                plain_features = apply_group_availability(
                    standardized[test], compact_groups, available, evaluation_centers, append_mask=False
                )
                for method, model, features, scaler in (
                    ("TCSS-mask-aware", mask_model, mask_features, mask_scaler),
                    ("unconditioned", unconditioned, plain_features, unconditioned_scaler),
                ):
                    raw_bundle = bundle_from_model(
                        model,
                        features,
                        y[test],
                        run_id=run_id,
                        dataset="primary",
                        method=method,
                        split="test",
                        seed=seed,
                        sample_ids=sample_ids[test],
                        date_ids=dates[test],
                        metadata={
                            "mask_id": record.mask_id,
                            "mask_strategy": strategy,
                            "unavailable_count": record.unavailable_count,
                            "selected_k": selected_k,
                        },
                    )
                    calibrated = PredictionBundle(
                        run_id=raw_bundle.run_id,
                        dataset=raw_bundle.dataset,
                        method=raw_bundle.method,
                        split=raw_bundle.split,
                        seed=raw_bundle.seed,
                        sample_ids=raw_bundle.sample_ids,
                        date_ids=raw_bundle.date_ids,
                        y_true=raw_bundle.y_true,
                        classes=raw_bundle.classes,
                        probabilities=scaler.predict_proba(raw_bundle.logits),
                        logits=raw_bundle.logits,
                        metadata={**(raw_bundle.metadata or {}), "temperature": scaler.temperature_},
                    )
                    calibrated.validate()
                    bundles.append(calibrated)
                mask_receipts.append(
                    {
                        "mask_id": record.mask_id,
                        "strategy": strategy,
                        "seed": seed,
                        "unavailable_count": record.unavailable_count,
                        "unavailable_groups": [str(value) for value in record.unavailable_groups],
                    }
                )
        tuning_records.extend({**row, "method": "TCSS-mask-aware", "seed": seed} for row in tuning)
        tuning_records.extend({**row, "method": "unconditioned", "seed": seed} for row in unconditioned_tuning)
    return bundles, mask_receipts, tuning_records
