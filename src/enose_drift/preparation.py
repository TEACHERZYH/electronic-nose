"""Portable feature preparation; no scheduler or account dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence
import numpy as np
import pandas as pd
from .anchoring import BlankAnchorAdjuster
from .data import iter_primary_raw_measurements
from .features import FeatureSchema, environmental_feature_columns, fit_feature_schema, group_feature_columns_by_sensor, response_feature_columns, transform_trace_features
from .primary_audit import PrimaryAuditExpectations, PrimaryAuditResult, run_primary_audit
@dataclass(frozen=True)
class FeatureVariant:
    name: str
    feature_names: tuple[str, ...]
    X_by_budget: Mapping[int, np.ndarray]
    groups: Mapping[Hashable, np.ndarray]
    always_include_columns: tuple[int, ...] = ()
    schema: FeatureSchema | None = None
    anchor_receipts: Mapping[int, tuple[Any, ...]] | None = None


@dataclass(frozen=True)
class PreparedPrimary:
    audit: PrimaryAuditResult
    scored_manifest: pd.DataFrame
    stage0_X: np.ndarray
    stage0_feature_names: tuple[str, ...]
    variants: Mapping[str, FeatureVariant]


def _expected_primary(config: Mapping[str, Any]) -> PrimaryAuditExpectations:
    data = config["data"]
    split = config["split"]
    return PrimaryAuditExpectations(
        expected_size_bytes=int(data["expected_size_bytes"]),
        expected_md5=str(data["expected_md5"]),
        expected_measurements=int(data["expected_measurements"]),
        expected_official_feature_rows=int(data["expected_official_feature_rows"]),
        expected_valid_dates=int(data["expected_valid_dates"]),
        expected_class_counts=dict(data["expected_class_counts"]),
        excluded_day_numbers=tuple(int(value) for value in data["excluded_day_labels"]),
        expected_sensors=int(data["expected_sensors"]),
        train_dates=int(split["train_dates"]),
        validation_dates=int(split["validation_dates"]),
        test_dates=int(split["test_dates"]),
        dates_per_environment=int(config["tcss"]["dates_per_environment"]),
        anchors_per_date=int(config["anchors"]["candidate_count_per_date"]),
        minimum_anchors_per_date=int(config["anchors"]["minimum_causal_count_per_date"]),
        expected_blank_anchor_count=int(config["anchors"]["expected_candidate_count"]),
        expected_scored_split_counts={
            str(key): int(value)
            for key, value in config["anchors"]["expected_scored_split_counts"].items()
        },
    )


def _groups_to_indices(feature_names: Sequence[str]) -> dict[int, np.ndarray]:
    named = group_feature_columns_by_sensor(feature_names)
    lookup = {name: index for index, name in enumerate(feature_names)}
    groups = {
        sensor: np.asarray([lookup[name] for name in columns], dtype=int)
        for sensor, columns in named.items()
    }
    if sorted(groups) != list(range(1, 63)):
        raise ValueError(f"Derived response features do not cover sensors 1..62: {sorted(groups)}")
    return groups


def _anchor_variants(
    X_full: np.ndarray,
    all_manifest: pd.DataFrame,
    groups: Mapping[Hashable, Sequence[int]],
    budgets: Sequence[int],
    *,
    allow_partial_budget_3: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[Any, ...]]]:
    candidate = all_manifest["is_blank_anchor"].to_numpy(dtype=bool)
    scored = ~candidate
    train_anchor = candidate & all_manifest["split"].eq("train").to_numpy()
    adjuster = BlankAnchorAdjuster(groups).fit(X_full, train_anchor)
    output: dict[int, np.ndarray] = {}
    receipts: dict[int, tuple[Any, ...]] = {}
    for budget in budgets:
        adjusted, receipt = adjuster.transform(
            X_full,
            all_manifest["date"].to_numpy(),
            pd.to_numeric(all_manifest["within_day_order"], errors="raise").to_numpy(dtype=float),
            candidate,
            scored,
            budget=int(budget),
            allow_partial_budget=bool(allow_partial_budget_3 and int(budget) == 3),
        )
        output[int(budget)] = adjusted[scored]
        receipts[int(budget)] = receipt
    return output, receipts


def _raw_feature_frame(
    audit: PrimaryAuditResult,
    config: Mapping[str, Any],
    *,
    representation: str,
) -> tuple[FeatureSchema, pd.DataFrame]:
    measurements = {
        measurement.sample_name: measurement.frame
        for measurement in iter_primary_raw_measurements(audit.source)
    }
    ordered_ids = audit.assigned_manifest["sample_id"].astype(str).tolist()
    missing = sorted(set(ordered_ids).difference(measurements))
    extra = sorted(set(measurements).difference(ordered_ids))
    if missing or extra:
        raise ValueError(f"Raw measurement alignment failed; missing={missing[:5]}, extra={extra[:5]}")
    training_ids = audit.assigned_manifest.loc[
        audit.assigned_manifest["split"].eq("train"), "sample_id"
    ].astype(str)
    training_items = [(sample_id, measurements[sample_id]) for sample_id in training_ids]
    schema = fit_feature_schema(
        training_items,
        explicit_stage_map=config["features"].get("cycle_stage_map"),
        representation=representation,
        include_environment=True,
        include_baseline_controls=bool(config["features"]["include_baseline_controls_for_stage0"]),
        standardize=False,
    )
    transformed = transform_trace_features(
        [(sample_id, measurements[sample_id]) for sample_id in ordered_ids], schema, standardize=False
    ).set_index("sample_id", verify_integrity=True)
    return schema, transformed.loc[ordered_ids]


def _build_raw_variant(
    audit: PrimaryAuditResult,
    schema: FeatureSchema,
    feature_frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    name: str,
    drop_recovery: bool = False,
    include_environment: bool | None = None,
) -> FeatureVariant:
    response_names = list(response_feature_columns(feature_frame.columns))
    if drop_recovery:
        response_names = [name for name in response_names if "recovery" not in name.casefold()]
    use_environment = (
        bool(config["features"]["include_temperature_humidity_in_primary"])
        if include_environment is None
        else include_environment
    )
    environment_names = list(environmental_feature_columns(feature_frame.columns)) if use_environment else []
    feature_names = tuple(response_names + environment_names)
    X_full = feature_frame.loc[:, feature_names].to_numpy(dtype=float)
    groups = _groups_to_indices(response_names)
    always = tuple(range(len(response_names), len(feature_names)))
    X_by_budget, anchor_receipts = _anchor_variants(
        X_full,
        audit.scoring.all_samples,
        groups,
        config["anchors"]["budgets"],
        allow_partial_budget_3=bool(config["anchors"].get("allow_partial_budget_3_on_short_days")),
    )
    return FeatureVariant(
        name,
        feature_names,
        X_by_budget,
        groups,
        always,
        schema,
        anchor_receipts,
    )


def _build_official_variant(audit: PrimaryAuditResult, config: Mapping[str, Any]) -> FeatureVariant:
    feature_names = tuple(
        column for sensor in sorted(audit.feature_groups) for column in audit.feature_groups[sensor]
    )
    X_full = audit.feature_table.loc[:, feature_names].to_numpy(dtype=float)
    lookup = {name: index for index, name in enumerate(feature_names)}
    groups = {
        sensor: np.asarray([lookup[name] for name in audit.feature_groups[sensor]], dtype=int)
        for sensor in sorted(audit.feature_groups)
    }
    X_by_budget, anchor_receipts = _anchor_variants(
        X_full,
        audit.scoring.all_samples,
        groups,
        config["anchors"]["budgets"],
        allow_partial_budget_3=bool(config["anchors"].get("allow_partial_budget_3_on_short_days")),
    )
    return FeatureVariant(
        "official_four_features",
        feature_names,
        X_by_budget,
        groups,
        anchor_receipts=anchor_receipts,
    )


def prepare_primary(
    archive_path: str | Path,
    config: Mapping[str, Any],
    *,
    include_sensitivity: bool = True,
) -> PreparedPrimary:
    audit = run_primary_audit(archive_path, expectations=_expected_primary(config))
    if not audit.passed:
        failed = [name for name, check in audit.checks.items() if not check.passed]
        raise ValueError(f"Primary structural audit did not pass: {failed}")
    primary_representation = str(config["features"]["primary_representation"])
    relative_schema, relative_frame = _raw_feature_frame(
        audit, config, representation=primary_representation
    )
    scored_ids = audit.scoring.scored["sample_id"].astype(str).tolist()
    stage0_names = tuple(relative_frame.columns.astype(str))
    stage0_X = relative_frame.loc[scored_ids, stage0_names].to_numpy(dtype=float)

    variants: dict[str, FeatureVariant] = {}
    variants["primary"] = _build_raw_variant(
        audit, relative_schema, relative_frame, config, name="primary"
    )
    variants["no_temperature_humidity"] = _build_raw_variant(
        audit,
        relative_schema,
        relative_frame,
        config,
        name="no_temperature_humidity",
        include_environment=False,
    )
    variants["no_recovery"] = _build_raw_variant(
        audit,
        relative_schema,
        relative_frame,
        config,
        name="no_recovery",
        drop_recovery=True,
    )
    if config["features"].get("official_feature_table_role") != "provenance_only":
        variants["official_four_features"] = _build_official_variant(audit, config)
    if include_sensitivity:
        for representation in config["features"].get("sensitivity_representations", []):
            schema, frame = _raw_feature_frame(audit, config, representation=str(representation))
            variants[f"representation_{representation}"] = _build_raw_variant(
                audit, schema, frame, config, name=f"representation_{representation}"
            )
    return PreparedPrimary(
        audit=audit,
        scored_manifest=audit.scoring.scored.copy(),
        stage0_X=stage0_X,
        stage0_feature_names=stage0_names,
        variants=variants,
    )


def _bundle_frames(bundles: Sequence) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    date_records: list[dict[str, Any]] = []
    for bundle in bundles:
        frame = bundle.sample_frame()
        for key, value in (bundle.metadata or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                frame[key] = value
            else:
                frame[key] = canonical_json_bytes(value).decode("utf-8")
        prediction_frames.append(frame)
        for record in bundle.date_records():
            flat = dict(record)
            for key in ("labels", "class_recall", "confusion_matrix"):
                flat[key] = canonical_json_bytes(flat[key]).decode("utf-8")
            for key, value in (bundle.metadata or {}).items():
                flat[key] = (
                    value
                    if isinstance(value, (str, int, float, bool)) or value is None
                    else canonical_json_bytes(value).decode("utf-8")
                )
            date_records.append(flat)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    return predictions, pd.DataFrame(date_records)


import dataclasses,json
def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON artifacts must not contain NaN or infinite values")
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for configuration and manifest hashes."""

    normalized = _json_ready(payload)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

