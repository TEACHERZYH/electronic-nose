"""Leakage-safe feature extraction from three-stage raw sensor traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_STAGES = ("baseline", "exposure", "recovery")


class StageMappingError(ValueError):
    """Raised when cycle-stage codes cannot be mapped without guessing."""


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _stage_key(value: object) -> str:
    return str(value).strip().casefold()


@dataclass(frozen=True)
class StageMap:
    """Explicit map from observed cycle-stage values to semantic stages."""

    raw_to_canonical: Mapping[str, str]

    def transform(self, values: pd.Series) -> pd.Series:
        transformed = values.map(lambda value: self.raw_to_canonical.get(_stage_key(value)))
        if transformed.isna().any():
            unknown = sorted(values.loc[transformed.isna()].astype(str).unique())
            raise StageMappingError(f"Unmapped Cycle_Stage values: {unknown}")
        return transformed.astype(str)

    def to_dict(self) -> dict[str, str]:
        return dict(self.raw_to_canonical)


@dataclass(frozen=True)
class FeatureSchema:
    """All quantities fitted exclusively from training measurements."""

    resistance_columns: tuple[str, ...]
    stage_column: str
    time_column: str
    stage_map: StageMap
    epsilon: float
    representation: str
    include_environment: bool
    include_baseline_controls: bool
    feature_columns: tuple[str, ...]
    impute_values: Mapping[str, float]
    center_values: Mapping[str, float]
    scale_values: Mapping[str, float]
    standardize: bool
    training_ids_sha256: str
    fit_scope: str = "training_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "resistance_columns": list(self.resistance_columns),
            "stage_column": self.stage_column,
            "time_column": self.time_column,
            "stage_map": self.stage_map.to_dict(),
            "epsilon": self.epsilon,
            "representation": self.representation,
            "include_environment": self.include_environment,
            "include_baseline_controls": self.include_baseline_controls,
            "feature_columns": list(self.feature_columns),
            "impute_values": dict(self.impute_values),
            "center_values": dict(self.center_values),
            "scale_values": dict(self.scale_values),
            "standardize": self.standardize,
            "training_ids_sha256": self.training_ids_sha256,
            "fit_scope": self.fit_scope,
        }


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(_normalise_name(candidate))
        if match is not None:
            return match
    return None


def find_stage_column(frame: pd.DataFrame) -> str:
    column = _find_column(frame, ("Cycle_Stage", "CycleStage", "Stage", "Phase"))
    if column is None:
        raise KeyError("Cycle_Stage")
    return column


def find_time_column(frame: pd.DataFrame) -> str:
    column = _find_column(frame, ("Time[hh:mm:ss]", "Time", "Timestamp", "ElapsedTime", "Seconds"))
    if column is None:
        raise KeyError("Time[hh:mm:ss]")
    return column


def resistance_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return raw R1--R62 columns in physical-sensor order."""

    matches: list[tuple[int, str]] = []
    for column in frame.columns:
        text = str(column).strip()
        match = re.fullmatch(r"R\s*0*(\d{1,2})(?:\s*\[\s*Ohm\s*\])?", text, flags=re.IGNORECASE)
        if match:
            sensor = int(match.group(1))
            if sensor > 0:
                matches.append((sensor, str(column)))
    if not matches:
        raise KeyError("No resistance columns matching R1[Ohm] ... R62[Ohm] were found")
    sensors = [sensor for sensor, _ in matches]
    if len(sensors) != len(set(sensors)):
        raise ValueError("Duplicate physical sensor number in raw resistance columns")
    return tuple(column for _, column in sorted(matches))


def _recognise_named_stage(raw_value: object) -> str | None:
    text = re.sub(r"[_\-]+", " ", str(raw_value).strip().casefold())
    if any(token in text for token in ("baseline", "base line", "pre exposure", "preexposure", "clean air")):
        return "baseline"
    if any(token in text for token in ("sample exposure", "exposure", "sample", "gas on")):
        return "exposure"
    if any(token in text for token in ("recovery", "recover", "purge", "washout")):
        return "recovery"
    return None


def _materialise_measurements(
    measurements: Mapping[str, pd.DataFrame] | Iterable[object],
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(measurements, Mapping):
        items = [(str(name), frame) for name, frame in measurements.items()]
    else:
        items: list[tuple[str, pd.DataFrame]] = []
        for index, item in enumerate(measurements):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], pd.DataFrame):
                name, frame = item
            elif hasattr(item, "sample_name") and hasattr(item, "frame"):
                name, frame = getattr(item, "sample_name"), getattr(item, "frame")
            else:
                raise TypeError(f"Measurement {index} is not (sample_id, DataFrame) or RawMeasurement-like")
            items.append((str(name), frame))
    if not items:
        raise ValueError("At least one measurement is required")
    return items


def fit_stage_map(
    measurements: Mapping[str, pd.DataFrame] | Iterable[object],
    *,
    stage_column: str | None = None,
    explicit_map: Mapping[object, str] | None = None,
) -> StageMap:
    """Fit a semantic stage map without guessing numeric codes.

    Named stages may be recognized directly.  Numeric or opaque official codes
    require a documented explicit map after their actual values are inspected.
    """

    items = _materialise_measurements(measurements)
    resolved_column = stage_column or find_stage_column(items[0][1])
    observed: list[object] = []
    for _, frame in items:
        if resolved_column not in frame:
            raise KeyError(resolved_column)
        observed.extend(frame[resolved_column].dropna().unique().tolist())
    unique = list(dict.fromkeys(_stage_key(value) for value in observed))

    if explicit_map is not None:
        mapping = {_stage_key(raw): str(canonical).strip().casefold() for raw, canonical in explicit_map.items()}
        invalid = sorted(set(mapping.values()).difference(CANONICAL_STAGES))
        if invalid:
            raise StageMappingError(f"Invalid canonical stage names: {invalid}")
        missing = sorted(set(unique).difference(mapping))
        if missing:
            raise StageMappingError(f"Explicit stage map does not cover observed values: {missing}")
    else:
        mapping = {}
        for raw in unique:
            canonical = _recognise_named_stage(raw)
            if canonical is None:
                raise StageMappingError(
                    "Opaque or numeric Cycle_Stage values require an explicit documented mapping; "
                    f"unrecognized value: {raw!r}"
                )
            mapping[raw] = canonical

    present = set(mapping[value] for value in unique)
    missing_stages = sorted(set(CANONICAL_STAGES).difference(present))
    if missing_stages:
        raise StageMappingError(f"Measurements do not contain all three required stages: {missing_stages}")
    return StageMap(raw_to_canonical=mapping)


def _elapsed_seconds(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        seconds = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    else:
        timedeltas = pd.to_timedelta(values.astype(str), errors="coerce")
        seconds = timedeltas.dt.total_seconds().to_numpy(dtype=float)
        if np.isnan(seconds).any():
            datetimes = pd.to_datetime(values, errors="coerce")
            if datetimes.notna().all():
                seconds = (datetimes - datetimes.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    if np.isnan(seconds).any():
        raise ValueError("Time values could not be parsed for every raw trace row")

    unwrapped = seconds.copy()
    offset = 0.0
    for index in range(1, len(unwrapped)):
        candidate = seconds[index] + offset
        if candidate < unwrapped[index - 1] and unwrapped[index - 1] - candidate > 12 * 3600:
            offset += 24 * 3600
            candidate = seconds[index] + offset
        unwrapped[index] = candidate
    if np.any(np.diff(unwrapped) < 0):
        raise ValueError("Raw trace time is not monotonic")
    return unwrapped - unwrapped[0]


def _linear_slope(time: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(time) & np.isfinite(values)
    x = time[finite]
    y = values[finite]
    if len(x) < 2 or np.ptp(x) <= 0:
        return float("nan")
    x_centered = x - x.mean()
    return float(np.dot(x_centered, y - y.mean()) / np.dot(x_centered, x_centered))


def _recovery_time_constant(time: np.ndarray, response: np.ndarray) -> float:
    finite = np.isfinite(time) & np.isfinite(response)
    x = time[finite]
    y = np.abs(response[finite])
    if len(x) < 2 or np.ptp(x) <= 0:
        return float("nan")
    x = x - x[0]
    initial_window = max(1, min(len(y), math.ceil(0.1 * len(y))))
    initial = float(np.nanmedian(y[:initial_window]))
    if not np.isfinite(initial) or initial <= np.finfo(float).eps:
        return 0.0
    threshold = initial / math.e
    crossings = np.flatnonzero(y <= threshold)
    if len(crossings):
        return float(x[crossings[0]])

    positive = y > max(initial * 1e-8, np.finfo(float).tiny)
    slope = _linear_slope(x[positive], np.log(y[positive]))
    if np.isfinite(slope) and slope < 0:
        return float(-1.0 / slope)
    # Right-censored proxy: recovery did not reach 1/e during observation.
    return float(x[-1])


def _sensor_number(column: str) -> int:
    match = re.match(r"R\s*0*(\d+)", str(column), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot determine sensor number from {column!r}")
    return int(match.group(1))


def _response(values: np.ndarray, baseline: float, epsilon: float, representation: str) -> np.ndarray:
    if representation == "relative":
        return (values - baseline) / (abs(baseline) + epsilon)
    if representation == "log":
        numerator = values + epsilon
        denominator = baseline + epsilon
        if denominator <= 0 or np.any(numerator <= 0):
            raise ValueError("Log-ratio representation requires positive resistance values")
        return np.log(numerator / denominator)
    raise ValueError("representation must be 'relative' or 'log'")


def extract_trace_features(
    trace: pd.DataFrame,
    *,
    stage_map: StageMap,
    epsilon: float,
    resistance_cols: Sequence[str] | None = None,
    stage_column: str | None = None,
    time_column: str | None = None,
    representation: str = "relative",
    include_environment: bool = True,
    include_baseline_controls: bool = False,
) -> pd.Series:
    """Extract one measurement's response descriptors using a fixed epsilon."""

    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a positive value fitted from training baselines")
    stage_col = stage_column or find_stage_column(trace)
    time_col = time_column or find_time_column(trace)
    sensor_cols = tuple(resistance_cols or resistance_columns(trace))
    missing = sorted(set(sensor_cols).difference(trace.columns))
    if missing:
        raise KeyError(f"Trace is missing fitted resistance columns: {missing}")

    canonical_stage = stage_map.transform(trace[stage_col])
    elapsed = _elapsed_seconds(trace[time_col])
    order = np.argsort(elapsed, kind="stable")
    canonical_stage = canonical_stage.iloc[order].reset_index(drop=True)
    elapsed = elapsed[order]
    ordered = trace.iloc[order].reset_index(drop=True)

    masks = {stage: canonical_stage.eq(stage).to_numpy() for stage in CANONICAL_STAGES}
    if any(mask.sum() == 0 for mask in masks.values()):
        raise ValueError("Every trace must contain baseline, exposure, and recovery rows")

    features: dict[str, float] = {}
    for column in sensor_cols:
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite resistance values in {column}")
        baseline = float(np.median(values[masks["baseline"]]))
        response = _response(values, baseline, epsilon, representation)
        exposure_time = elapsed[masks["exposure"]]
        exposure_time = exposure_time - exposure_time[0]
        exposure_response = response[masks["exposure"]]
        recovery_time = elapsed[masks["recovery"]]
        recovery_time = recovery_time - recovery_time[0]
        recovery_response = response[masks["recovery"]]

        peak_index = int(np.argmax(np.abs(exposure_response)))
        initial_count = max(2, min(len(exposure_response), math.ceil(0.2 * len(exposure_response))))
        sensor_prefix = f"s{_sensor_number(column):02d}__"
        features[sensor_prefix + "amplitude"] = float(exposure_response[peak_index])
        features[sensor_prefix + "initial_slope"] = _linear_slope(
            exposure_time[:initial_count], exposure_response[:initial_count]
        )
        features[sensor_prefix + "overall_slope"] = _linear_slope(exposure_time, exposure_response)
        features[sensor_prefix + "auc"] = float(np.trapezoid(exposure_response, exposure_time))
        features[sensor_prefix + "time_to_peak_s"] = float(exposure_time[peak_index])
        features[sensor_prefix + "recovery_rate"] = _linear_slope(recovery_time, recovery_response)
        features[sensor_prefix + "recovery_tau_s"] = _recovery_time_constant(
            recovery_time, recovery_response
        )
        if include_baseline_controls:
            features[sensor_prefix + "baseline_median"] = baseline

    if include_environment:
        environmental = {
            "temperature": _find_column(ordered, ("T01[degC]", "Temperature", "Temp")),
            "humidity": _find_column(ordered, ("H01[%rh]", "Humidity", "RelativeHumidity", "RH")),
        }
        for quantity, column in environmental.items():
            if column is None:
                continue
            values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
            for stage in CANONICAL_STAGES:
                features[f"env__{quantity}__{stage}_mean"] = float(np.nanmean(values[masks[stage]]))
    return pd.Series(features, dtype=float)


def _baseline_values(
    trace: pd.DataFrame,
    *,
    stage_map: StageMap,
    stage_column: str,
    sensor_cols: Sequence[str],
) -> list[float]:
    stages = stage_map.transform(trace[stage_column])
    mask = stages.eq("baseline")
    if not mask.any():
        raise ValueError("Training trace has no baseline rows")
    values: list[float] = []
    for column in sensor_cols:
        sensor_values = pd.to_numeric(trace.loc[mask, column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sensor_values).all():
            raise ValueError(f"Non-finite baseline resistance in {column}")
        values.append(float(np.median(sensor_values)))
    return values


def fit_feature_schema(
    training_measurements: Mapping[str, pd.DataFrame] | Iterable[object],
    *,
    stage_map: StageMap | None = None,
    explicit_stage_map: Mapping[object, str] | None = None,
    representation: str = "relative",
    include_environment: bool = True,
    include_baseline_controls: bool = False,
    standardize: bool = True,
) -> FeatureSchema:
    """Fit epsilon, imputation, centering, and scaling on training traces only."""

    items = _materialise_measurements(training_measurements)
    first = items[0][1]
    stage_col = find_stage_column(first)
    time_col = find_time_column(first)
    sensor_cols = resistance_columns(first)
    resolved_stage_map = stage_map or fit_stage_map(
        items, stage_column=stage_col, explicit_map=explicit_stage_map
    )

    baselines: list[float] = []
    for _, trace in items:
        if find_stage_column(trace) != stage_col or find_time_column(trace) != time_col:
            raise ValueError("All raw traces must use a common stage/time schema")
        if resistance_columns(trace) != sensor_cols:
            raise ValueError("All raw traces must contain the same physical sensor columns")
        baselines.extend(
            _baseline_values(
                trace,
                stage_map=resolved_stage_map,
                stage_column=stage_col,
                sensor_cols=sensor_cols,
            )
        )
    median_abs_baseline = float(np.median(np.abs(np.asarray(baselines, dtype=float))))
    epsilon = 1e-8 * median_abs_baseline
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("Training baselines cannot define the required positive epsilon")

    raw_rows = []
    for sample_id, trace in items:
        row = extract_trace_features(
            trace,
            stage_map=resolved_stage_map,
            epsilon=epsilon,
            resistance_cols=sensor_cols,
            stage_column=stage_col,
            time_column=time_col,
            representation=representation,
            include_environment=include_environment,
            include_baseline_controls=include_baseline_controls,
        )
        row.name = sample_id
        raw_rows.append(row)
    raw = pd.DataFrame(raw_rows).replace([np.inf, -np.inf], np.nan)
    feature_columns = tuple(raw.columns.astype(str))

    impute: dict[str, float] = {}
    for column in feature_columns:
        median = float(raw[column].median(skipna=True))
        impute[column] = median if np.isfinite(median) else 0.0
    filled = raw.fillna(impute)
    center = {column: float(filled[column].mean()) for column in feature_columns}
    scale = {}
    for column in feature_columns:
        value = float(filled[column].std(ddof=0))
        scale[column] = value if np.isfinite(value) and value > 1e-12 else 1.0

    training_hash = hashlib.sha256("\n".join(sample_id for sample_id, _ in items).encode("utf-8")).hexdigest()
    return FeatureSchema(
        resistance_columns=sensor_cols,
        stage_column=stage_col,
        time_column=time_col,
        stage_map=resolved_stage_map,
        epsilon=epsilon,
        representation=representation,
        include_environment=include_environment,
        include_baseline_controls=include_baseline_controls,
        feature_columns=feature_columns,
        impute_values=impute,
        center_values=center,
        scale_values=scale,
        standardize=standardize,
        training_ids_sha256=training_hash,
    )


def transform_trace_features(
    measurements: Mapping[str, pd.DataFrame] | Iterable[object],
    schema: FeatureSchema,
    *,
    standardize: bool | None = None,
) -> pd.DataFrame:
    """Transform new traces without updating any training-derived quantity."""

    items = _materialise_measurements(measurements)
    rows = []
    for sample_id, trace in items:
        row = extract_trace_features(
            trace,
            stage_map=schema.stage_map,
            epsilon=schema.epsilon,
            resistance_cols=schema.resistance_columns,
            stage_column=schema.stage_column,
            time_column=schema.time_column,
            representation=schema.representation,
            include_environment=schema.include_environment,
            include_baseline_controls=schema.include_baseline_controls,
        )
        row.name = sample_id
        rows.append(row)
    raw = pd.DataFrame(rows).reindex(columns=schema.feature_columns)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(dict(schema.impute_values))
    use_standardization = schema.standardize if standardize is None else standardize
    if use_standardization:
        for column in schema.feature_columns:
            raw[column] = (raw[column] - schema.center_values[column]) / schema.scale_values[column]
    result = raw.reset_index(names="sample_id")
    result.attrs["fit_scope"] = schema.fit_scope
    result.attrs["training_ids_sha256"] = schema.training_ids_sha256
    result.attrs["feature_columns"] = list(schema.feature_columns)
    return result


def fit_transform_trace_features(
    training_measurements: Mapping[str, pd.DataFrame] | Iterable[object],
    **kwargs: object,
) -> tuple[FeatureSchema, pd.DataFrame]:
    """Fit a training-only schema and transform the same training measurements."""

    items = _materialise_measurements(training_measurements)
    schema = fit_feature_schema(items, **kwargs)
    return schema, transform_trace_features(items, schema)


def group_feature_columns_by_sensor(columns: Sequence[str]) -> dict[int, tuple[str, ...]]:
    """Group derived or official feature names by physical sensor number."""

    groups: dict[int, list[str]] = {}
    for column in columns:
        text = str(column)
        derived = re.match(r"s0*(\d{1,2})__", text, flags=re.IGNORECASE)
        official = re.search(
            r"(?:^|[^a-z0-9])(?:r|sensor)\s*0*(\d{1,2})(?:[^0-9]|$)",
            text,
            flags=re.IGNORECASE,
        )
        match = derived or official
        if match:
            sensor = int(match.group(1))
            groups.setdefault(sensor, []).append(text)
    return {sensor: tuple(values) for sensor, values in sorted(groups.items())}


def response_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Select sensor-response descriptors while excluding audit-only baselines."""

    return tuple(
        str(column)
        for column in columns
        if re.match(r"s\d+__", str(column), flags=re.IGNORECASE)
        and not str(column).casefold().endswith("__baseline_median")
    )


def environmental_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(column) for column in columns if str(column).casefold().startswith("env__"))
