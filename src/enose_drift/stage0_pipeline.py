"""Execution of the frozen Stage-0 acquisition and leakage controls."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .baselines import tune_and_fit_probabilistic_model
from .controls import (
    identify_feature_families,
    random_row_partition,
    run_feature_control,
    run_ordinal_control,
)
from .evaluation import PredictionBundle, bundle_from_model


def _split_indices(manifest: pd.DataFrame):
    return tuple(np.flatnonzero(manifest["split"].eq(name).to_numpy()) for name in ("train", "validation", "test"))


def _run_position_model(
    manifest: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    run_id: str,
    seed: int,
) -> tuple[PredictionBundle, list[dict]]:
    order_col = "observed_ordinal" if "observed_ordinal" in manifest else "within_day_order"
    order = pd.to_numeric(manifest[order_col], errors="raise").to_numpy(dtype=float)
    day_size = manifest.groupby("date")[order_col].transform("max").to_numpy(dtype=float)
    relative = order / np.maximum(day_size, 1.0)
    X = np.column_stack([order, relative])
    y = manifest["label"].to_numpy()
    model, tuning = tune_and_fit_probabilistic_model(
        "rbf_svm", X[train], y[train], X[validation], y[validation], seed=seed
    )
    bundle = bundle_from_model(
        model,
        X[test],
        y[test],
        run_id=run_id,
        dataset="primary",
        method="position_only",
        split="test",
        seed=seed,
        sample_ids=manifest.iloc[test]["sample_id"].to_numpy(),
        date_ids=manifest.iloc[test]["date"].to_numpy(),
        metadata={"features": [order_col, "relative_within_day_position"], "control_only": True},
    )
    return bundle, tuning


def run_stage0(
    X: np.ndarray,
    feature_names: Sequence[str],
    scored_manifest: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    run_id: str,
    permutation_repeats: int | None = None,
) -> tuple[list[PredictionBundle], list[dict], list[dict]]:
    """Run every Stage-0 branch and retain unavailable-family states explicitly."""

    train, validation, test = _split_indices(scored_manifest)
    bundles: list[PredictionBundle] = []
    tuning_records: list[dict] = []
    control_status: list[dict] = []
    seeds = [int(value) for value in config["tcss"]["seeds"]]
    families = identify_feature_families(feature_names)
    family_sizes = {
        "full_response": len(families.full_response),
        "baseline_only": len(families.baseline),
        "temperature_humidity_only": len(families.temperature_humidity),
        "recovery_history_only": len(families.recovery_history),
    }
    random_row_config = config.get("stage0", {}).get("random_row", {})
    coverage_split = str(random_row_config.get("required_coverage_split", "test"))
    minimum_per_group = int(random_row_config.get("minimum_per_date_class", 1))
    label_column = "label" if "label" in scored_manifest.columns else "class"
    required_mask = scored_manifest["split"].astype(str).eq(coverage_split)
    required_test_groups = np.full(len(scored_manifest), None, dtype=object)
    required_test_groups[required_mask.to_numpy()] = (
        scored_manifest.loc[required_mask, "date"].astype(str)
        + "\x1f"
        + scored_manifest.loc[required_mask, label_column].astype(str)
    ).to_numpy()
    expected_random_groups = {
        str(value) for value in required_test_groups if value is not None and str(value).strip()
    }
    target_partition_sizes = tuple(
        int(scored_manifest["split"].astype(str).eq(name).sum())
        for name in ("train", "validation", "test")
    )

    for seed in seeds:
        ordinal = run_ordinal_control(scored_manifest, train, test, run_id=run_id, seed=seed)
        bundles.append(ordinal)
        position, tuning = _run_position_model(
            scored_manifest, train, validation, test, run_id=run_id, seed=seed
        )
        bundles.append(position)
        tuning_records.extend({**row, "control": "position_only", "seed": seed} for row in tuning)

        for control in ("full_response", "baseline_only", "temperature_humidity_only", "recovery_history_only"):
            if family_sizes[control] == 0:
                control_status.append(
                    {
                        "control": control,
                        "seed": seed,
                        "status": "WAITING_RESOURCE",
                        "reason": "required feature family is absent from the supplied representation",
                    }
                )
                continue
            validation_bundle, test_bundle, tuning = run_feature_control(
                control,
                X,
                scored_manifest,
                feature_names,
                train,
                validation,
                test,
                run_id=run_id,
                seed=seed,
            )
            bundles.extend([validation_bundle, test_bundle])
            tuning_records.extend({**row, "control": control, "seed": seed} for row in tuning)
            control_status.append({"control": control, "seed": seed, "status": "VERIFIED_EXECUTION"})

        random_train, random_validation, random_test = random_row_partition(
            scored_manifest[label_column].to_numpy(),
            seed=seed,
            required_test_groups=required_test_groups,
            minimum_per_required_group=minimum_per_group,
            target_partition_sizes=target_partition_sizes,
        )
        covered_random_groups = {
            str(value)
            for value in required_test_groups[random_test]
            if value is not None and str(value).strip()
        }
        if covered_random_groups != expected_random_groups:
            raise RuntimeError("random-row coverage receipt does not match the frozen date-by-class contract")
        random_validation_bundle, random_test_bundle, tuning = run_feature_control(
            "random_row",
            X,
            scored_manifest,
            feature_names,
            random_train,
            random_validation,
            random_test,
            run_id=run_id,
            seed=seed,
        )
        bundles.extend([random_validation_bundle, random_test_bundle])
        tuning_records.extend({**row, "control": "random_row", "seed": seed} for row in tuning)
        control_status.append(
            {
                "control": "random_row",
                "seed": seed,
                "status": "VERIFIED_EXECUTION",
                "train_rows": len(random_train),
                "validation_rows": len(random_validation),
                "test_rows": len(random_test),
                "required_coverage_split": coverage_split,
                "required_date_class_groups": len(expected_random_groups),
                "covered_date_class_groups": len(covered_random_groups),
                "minimum_per_date_class": minimum_per_group,
            }
        )

    repeats = int(
        config["stage0"]["permutation_repeats"] if permutation_repeats is None else permutation_repeats
    )
    for repeat in range(repeats):
        seed = int(np.random.SeedSequence([seeds[0], repeat]).generate_state(1)[0])
        _, test_bundle, tuning = run_feature_control(
            "clustered_permutation",
            X,
            scored_manifest,
            feature_names,
            train,
            validation,
            test,
            run_id=run_id,
            seed=seed,
            permute_training_labels=True,
        )
        bundles.append(test_bundle)
        tuning_records.extend(
            {**row, "control": "clustered_permutation", "permutation_repeat": repeat, "seed": seed}
            for row in tuning
        )
    return bundles, tuning_records, control_status


def evaluate_stage0_gate(bundles: Sequence[PredictionBundle], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen v1.1 quantitative gate without inspecting later results."""

    test_bundles = [bundle for bundle in bundles if bundle.split == "test"]
    method_seed_date: dict[str, dict[int, dict[str, float]]] = {}
    permutation_means: list[float] = []
    for bundle in test_bundles:
        date_mapping = {
            str(record["date_id"]): float(record["macro_f1"]) for record in bundle.date_records()
        }
        if bundle.method == "clustered_permutation":
            permutation_means.append(float(np.mean(list(date_mapping.values()))))
            continue
        method_seed_date.setdefault(bundle.method, {})[int(bundle.seed)] = date_mapping

    method_means: dict[str, float] = {}
    method_date_means: dict[str, dict[str, float]] = {}
    method_date_seed_counts: dict[str, dict[str, int]] = {}
    for method, by_seed in method_seed_date.items():
        common_dates = sorted(set.intersection(*(set(values) for values in by_seed.values())))
        date_means = {
            date: float(np.mean([by_seed[seed][date] for seed in sorted(by_seed)])) for date in common_dates
        }
        method_date_means[method] = date_means
        method_date_seed_counts[method] = {date: len(by_seed) for date in common_dates}
        method_means[method] = float(np.mean(list(date_means.values())))

    if "full_response" not in method_means:
        raise ValueError("Stage-0 gate requires full_response test predictions.")
    if not permutation_means:
        raise ValueError("Stage-0 gate requires clustered-permutation test predictions.")
    gate = config["stage0"]["gate"]
    null_upper = float(np.quantile(permutation_means, float(gate["permutation_upper_quantile"])))
    full_mean = method_means["full_response"]
    learnable_signal = bool(full_mean > null_upper)
    proxy_names = (
        "observed_ordinal",
        "position_only",
        "baseline_only",
        "temperature_humidity_only",
        "recovery_history_only",
    )
    available_proxy = {name: method_means[name] for name in proxy_names if name in method_means}
    margin = float(gate["proxy_near_margin_macro_f1"])
    near_proxy = {
        name: score for name, score in available_proxy.items() if score >= full_mean - margin
    }
    if not learnable_signal:
        branch = gate["branches"]["audit_only"]
    elif near_proxy:
        branch = gate["branches"]["protocol_conditioned"]
    else:
        branch = gate["branches"]["response"]
    return {
        "gate_version": "v1.1",
        "full_response_date_mean_macro_f1": full_mean,
        "permutation_null_upper": null_upper,
        "permutation_repeats": len(permutation_means),
        "learnable_signal": learnable_signal,
        "proxy_near_margin_macro_f1": margin,
        "proxy_date_mean_macro_f1": available_proxy,
        "near_or_higher_proxies": near_proxy,
        "branch": branch,
        "method_date_mean_macro_f1": method_date_means,
        "method_date_seed_counts": method_date_seed_counts,
    }
