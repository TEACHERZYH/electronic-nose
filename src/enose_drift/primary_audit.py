"""Formal structural audit for the one-year 62-sensor primary dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from .data import (
    PrimarySource,
    build_primary_manifest,
    discover_primary_source,
    load_primary_features,
    parse_primary_label,
    read_delimited_auto,
)
from .features import group_feature_columns_by_sensor
from .protocol import (
    PrimaryDateSplit,
    ScoringSets,
    assign_primary_split,
    build_scoring_sets,
    build_training_environments,
    chronological_date_split,
    identify_blank_anchors,
)


AUDIT_SCHEMA_VERSION = "enose-primary-audit/v1"


def _default_class_counts() -> dict[str, int]:
    return {
        "EtOH": 232,
        "Diacetyl 0.1 ppm": 117,
        "Diacetyl 1 ppm": 117,
        "2-Phenylethanol 200 ppm": 117,
        "2-Phenylethanol 1000 ppm": 117,
    }


@dataclass(frozen=True)
class PrimaryAuditExpectations:
    expected_size_bytes: int = 141_110_296
    expected_md5: str = "9937678cac4118c53287276009172a74"
    expected_sha256: str | None = None
    expected_measurements: int = 700
    expected_official_feature_rows: int | None = None
    expected_valid_dates: int = 39
    expected_class_counts: Mapping[str, int] = field(default_factory=_default_class_counts)
    excluded_day_numbers: tuple[int, ...] = (14,)
    short_day_counts: Mapping[int, int] = field(default_factory=lambda: {12: 17, 37: 17})
    regular_day_count: int = 18
    expected_sensors: int = 62
    expected_features_per_sensor: int = 4
    train_dates: int = 24
    validation_dates: int = 7
    test_dates: int = 8
    dates_per_environment: int = 4
    anchors_per_date: int = 3
    minimum_anchors_per_date: int = 3
    expected_blank_anchor_count: int | None = None
    expected_scored_split_counts: Mapping[str, int] = field(
        default_factory=lambda: {"train": 359, "validation": 105, "test": 119}
    )


@dataclass(frozen=True)
class ArchiveVerification:
    path: str
    size_bytes: int
    md5: str
    sha256: str
    expected_size_bytes: int
    expected_md5: str
    expected_sha256: str | None
    size_ok: bool
    md5_ok: bool
    sha256_ok: bool

    @property
    def passed(self) -> bool:
        return self.size_ok and self.md5_ok and self.sha256_ok

    def to_record(self) -> dict[str, Any]:
        return asdict(self) | {"passed": self.passed}


@dataclass(frozen=True)
class AuditCheck:
    passed: bool
    actual: Any
    expected: Any
    detail: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "actual": _json_ready(self.actual),
            "expected": _json_ready(self.expected),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PrimaryAuditResult:
    status: str
    debug: bool
    archive: ArchiveVerification
    source: PrimarySource
    expectations: PrimaryAuditExpectations
    feature_table: pd.DataFrame
    manifest: pd.DataFrame
    assigned_manifest: pd.DataFrame
    split: PrimaryDateSplit
    scoring: ScoringSets
    feature_groups: Mapping[int, tuple[str, ...]]
    checks: Mapping[str, AuditCheck]
    alignment_mode: str
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks.values())

    def to_record(self) -> dict[str, Any]:
        class_counts = self.manifest["label"].value_counts().sort_index().to_dict()
        split_counts = self.assigned_manifest["split"].value_counts().sort_index().to_dict()
        scored_split_counts = self.scoring.scored["split"].value_counts().sort_index().to_dict()
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": self.status,
            "debug": self.debug,
            "passed": self.passed,
            "archive": self.archive.to_record(),
            "source": {
                "root": str(self.source.root),
                "archive_path": None if self.source.archive_path is None else str(self.source.archive_path),
                "extracted_features_path": (
                    None
                    if self.source.extracted_features_path is None
                    else str(self.source.extracted_features_path)
                ),
                "extracted_features_member": self.source.extracted_features_member,
                "raw_measurement_count": self.source.raw_measurement_count,
            },
            "expectations": _json_ready(asdict(self.expectations)),
            "counts": {
                "measurements": len(self.manifest),
                "valid_dates": int(self.manifest["date"].nunique()),
                "class_counts": _json_ready(class_counts),
                "split_rows": _json_ready(split_counts),
                "candidate_anchors": len(self.scoring.anchors),
                "scored_samples": len(self.scoring.scored),
                "scored_split_rows": _json_ready(scored_split_counts),
                "sensor_groups": len(self.feature_groups),
                "official_sensor_features": int(sum(len(values) for values in self.feature_groups.values())),
            },
            "date_split": self.split.as_dict(),
            "alignment_mode": self.alignment_mode,
            "feature_groups": {
                str(sensor): list(columns) for sensor, columns in sorted(self.feature_groups.items())
            },
            "checks": {name: check.to_record() for name, check in self.checks.items()},
            "warnings": list(self.warnings),
        }

    def write_record(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_record(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return target


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_ready(item) for item in value]
    return value


def verify_primary_archive(
    archive_path: str | Path,
    expectations: PrimaryAuditExpectations = PrimaryAuditExpectations(),
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> ArchiveVerification:
    """Stream the archive once to verify its size, MD5, and SHA-256."""

    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            md5.update(chunk)
            sha256.update(chunk)
    actual_md5 = md5.hexdigest()
    actual_sha256 = sha256.hexdigest()
    expected_md5 = expectations.expected_md5.casefold()
    expected_sha256 = (
        None if expectations.expected_sha256 is None else expectations.expected_sha256.casefold()
    )
    return ArchiveVerification(
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        md5=actual_md5,
        sha256=actual_sha256,
        expected_size_bytes=expectations.expected_size_bytes,
        expected_md5=expected_md5,
        expected_sha256=expected_sha256,
        size_ok=path.stat().st_size == expectations.expected_size_bytes,
        md5_ok=actual_md5 == expected_md5,
        sha256_ok=expected_sha256 is None or actual_sha256 == expected_sha256,
    )


def _normalise_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {_normalise_column(column): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(_normalise_column(candidate))
        if match is not None:
            return match
    return None


def _day_number(value: object) -> int | None:
    match = re.search(r"(?:^|[^a-z])day[\s_-]*0*(\d+)(?:[^0-9]|$)", str(value), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _ethanol_replicate(value: object) -> int | None:
    match = re.search(r"(?:etoh|ethanol)[^0-9]{0,12}([1-6])(?:[^0-9]|$)", str(value), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_observed_date(value: object) -> str:
    text = str(value).strip()
    for pattern, date_format in (
        (r"\d{2}-\d{2}-\d{2}", "%y-%m-%d"),
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    ):
        if re.fullmatch(pattern, text):
            return datetime.strptime(text, date_format).date().isoformat()
    parsed = pd.to_datetime(text, errors="raise")
    return parsed.date().isoformat()


def _parse_observed_time(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    parsed = pd.to_timedelta(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Cannot parse raw measurement start time {value!r}")
    return float(parsed.total_seconds())


def _read_raw_header(source: PrimarySource, name: str) -> pd.DataFrame:
    if source.archive_path is not None and name in source.raw_csv_members:
        with zipfile.ZipFile(source.archive_path) as archive:
            return read_delimited_auto(archive.read(name), nrows=1)
    path = Path(name)
    return read_delimited_auto(path, nrows=1)


def _raw_header_manifest(source: PrimarySource) -> pd.DataFrame:
    entries: list[tuple[str, str]] = []
    if source.raw_csv_paths:
        entries.extend((str(path), str(path.relative_to(source.root))) for path in source.raw_csv_paths)
    if source.raw_csv_members:
        entries.extend((member, member) for member in source.raw_csv_members)
    if not entries:
        raise FileNotFoundError("Raw measurement CSV files are required to verify observed order")

    rows: list[dict[str, Any]] = []
    for access_name, source_name in entries:
        header = _read_raw_header(source, access_name)
        date_col = _find_column(header, ("Date[yyyy-mm-dd]", "Date", "MeasurementDate"))
        time_col = _find_column(header, ("Time[hh:mm:ss]", "Time", "Timestamp"))
        if date_col is None or time_col is None:
            raise ValueError(f"Raw CSV {source_name!r} lacks its documented date/time columns")
        basename = PurePosixPath(source_name.replace("\\", "/")).name
        rows.append(
            {
                "raw_source_name": source_name,
                "raw_basename": basename,
                "date": _parse_observed_date(header.iloc[0][date_col]),
                "day_number": _day_number(source_name),
                "label": parse_primary_label(basename),
                "ethanol_replicate": _ethanol_replicate(basename),
                "start_seconds": _parse_observed_time(header.iloc[0][time_col]),
            }
        )
    raw = pd.DataFrame(rows)
    if raw["day_number"].isna().any():
        raise ValueError("Could not parse a documented Day number from every raw member path")
    if raw.duplicated(["date", "start_seconds"]).any():
        duplicates = raw.loc[raw.duplicated(["date", "start_seconds"], keep=False)]
        raise ValueError(
            "Observed raw start times are not unique within a date: "
            + ", ".join(duplicates["raw_source_name"].head(6).astype(str))
        )
    raw["within_day_order"] = (
        raw.groupby("date")["start_seconds"].rank(method="first").astype(int)
    )
    return raw.sort_values(["date", "within_day_order"], kind="stable").reset_index(drop=True)


def _manifest_source_basename(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or re.fullmatch(r"sample_\d+", text):
        return None
    return PurePosixPath(text.replace("\\", "/")).name.casefold()


def _enrich_manifest_from_raw_headers(
    manifest: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    result = manifest.copy()
    result["_basename"] = result["source_name"].map(_manifest_source_basename)
    raw = raw.copy()
    raw["_basename"] = raw["raw_basename"].str.casefold()
    warnings: list[str] = []

    exact_possible = result["_basename"].notna().all() and result["day_number"].notna().all()
    if exact_possible:
        left_keys = list(zip(result["day_number"].astype(int), result["_basename"]))
        right = raw.set_index([raw["day_number"].astype(int), "_basename"], verify_integrity=True)
        if all(key in right.index for key in left_keys):
            matched = right.loc[left_keys].reset_index(drop=True)
            alignment_mode = "raw_exact_day_and_filename"
        else:
            exact_possible = False

    if not exact_possible:
        if result["day_number"].notna().all():
            identity_keys = ["day_number", "label"]
            order_keys = ["day_number", "within_day_order"]
            identity_description = "Day, canonical class, and within-class occurrence"
        elif result["date"].notna().all():
            # The official extracted_features.csv exposes an actual date but no
            # filename or Day label.  Date+class counts are checked one-to-one,
            # and raw members are ordered by their observed start time before
            # assigning the within-class occurrence.
            identity_keys = ["date", "label"]
            order_keys = ["date", "within_day_order"]
            identity_description = "actual date, canonical class, and within-class occurrence"
        else:
            raise ValueError("Feature manifest lacks both Day and actual-date identity for safe raw-header alignment")
        result["_occurrence"] = result.groupby(identity_keys, sort=False).cumcount()
        raw = raw.sort_values(order_keys, kind="stable")
        raw["_occurrence"] = raw.groupby(identity_keys, sort=False).cumcount()
        left_counts = result.groupby(identity_keys).size().sort_index()
        right_counts = raw.groupby(identity_keys).size().sort_index()
        if not left_counts.equals(right_counts):
            raise ValueError("Feature-table and raw-file date/class counts do not support safe alignment")
        matched = result[[*identity_keys, "_occurrence"]].merge(
            raw,
            on=[*identity_keys, "_occurrence"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        alignment_mode = (
            "raw_day_class_occurrence"
            if identity_keys[0] == "day_number"
            else "raw_date_class_occurrence"
        )
        warnings.append(
            f"Raw headers were aligned by {identity_description} because "
            "the feature table did not expose an exact filename key."
        )

    for column, raw_column in (
        ("date", "date"),
        ("day_number", "day_number"),
        ("within_day_order", "within_day_order"),
        ("ethanol_replicate", "ethanol_replicate"),
        ("source_name", "raw_source_name"),
    ):
        result[column] = matched[raw_column].to_numpy()
    result["sample_id"] = matched["raw_source_name"].astype(str).to_numpy()
    result = result.drop(columns=[column for column in ("_basename", "_occurrence") if column in result])
    if result["sample_id"].duplicated().any() or result["within_day_order"].isna().any():
        raise ValueError("Raw-header enrichment did not produce unique, fully ordered samples")
    return result, alignment_mode, tuple(warnings)


def identify_official_sensor_feature_groups(
    feature_table: pd.DataFrame,
    *,
    expected_sensors: int = 62,
    expected_features_per_sensor: int = 4,
) -> dict[int, tuple[str, ...]]:
    """Find numeric official feature columns and group them by physical sensor."""

    candidates = group_feature_columns_by_sensor([str(column) for column in feature_table.columns])
    groups: dict[int, tuple[str, ...]] = {}
    for sensor, columns in candidates.items():
        numeric = tuple(
            column
            for column in columns
            if column in feature_table and pd.api.types.is_numeric_dtype(feature_table[column])
        )
        if numeric:
            groups[sensor] = numeric
    expected_ids = set(range(1, expected_sensors + 1))
    if set(groups) != expected_ids:
        missing = sorted(expected_ids.difference(groups))
        extra = sorted(set(groups).difference(expected_ids))
        raise ValueError(f"Official feature groups do not match sensors 1..{expected_sensors}; missing={missing}, extra={extra}")
    wrong_sizes = {sensor: len(columns) for sensor, columns in groups.items() if len(columns) != expected_features_per_sensor}
    if wrong_sizes:
        raise ValueError(
            f"Expected {expected_features_per_sensor} official numeric features per sensor; found {wrong_sizes}"
        )
    return groups


def _check(actual: Any, expected: Any, *, detail: str = "") -> AuditCheck:
    return AuditCheck(passed=_json_ready(actual) == _json_ready(expected), actual=actual, expected=expected, detail=detail)


def run_primary_audit(
    archive_path: str | Path,
    *,
    expectations: PrimaryAuditExpectations = PrimaryAuditExpectations(),
    debug: bool = False,
) -> PrimaryAuditResult:
    """Run the frozen structural audit and return dataframes plus a JSON record."""

    archive = verify_primary_archive(archive_path, expectations)
    source = discover_primary_source(archive_path)
    feature_table = load_primary_features(source)
    raw_headers = _raw_header_manifest(source)
    manifest = raw_headers.rename(columns={"raw_source_name": "source_name"}).copy()
    manifest["sample_id"] = manifest["source_name"].astype(str)
    manifest["day_number"] = manifest["day_number"].astype("Int64")
    manifest["within_day_order"] = manifest["within_day_order"].astype("Int64")
    manifest["ethanol_replicate"] = manifest["ethanol_replicate"].astype("Int64")
    manifest["observed_ordinal"] = manifest["within_day_order"]
    manifest = manifest[
        [
            "sample_id",
            "date",
            "day_number",
            "label",
            "within_day_order",
            "observed_ordinal",
            "ethanol_replicate",
            "source_name",
        ]
    ].reset_index(drop=True)
    warnings_list: list[str] = []
    if len(feature_table) == len(manifest):
        feature_manifest = build_primary_manifest(feature_table)
        label_col = _find_column(feature_table, ("Class", "Label", "Gas", "Analyte"))
        if label_col is not None:
            feature_manifest["label"] = feature_table[label_col].map(parse_primary_label).to_numpy()
        _, alignment_mode, alignment_warnings = _enrich_manifest_from_raw_headers(
            feature_manifest, raw_headers
        )
        warnings_list.extend(alignment_warnings)
    else:
        alignment_mode = "raw_header_authoritative_feature_table_provenance_only"
        warnings_list.append(
            "The official feature table is a documented subset and has no one-to-one identity key; "
            "the 700 raw measurement files are authoritative for the formal experiment."
        )
    warnings = tuple(warnings_list)
    feature_groups = identify_official_sensor_feature_groups(
        feature_table,
        expected_sensors=expectations.expected_sensors,
        expected_features_per_sensor=expectations.expected_features_per_sensor,
    )

    split = chronological_date_split(
        manifest,
        n_train=expectations.train_dates,
        n_validation=expectations.validation_dates,
        n_test=expectations.test_dates,
    )
    assigned = assign_primary_split(manifest, split)
    assigned = build_training_environments(
        assigned,
        dates_per_environment=expectations.dates_per_environment,
    )
    anchors = identify_blank_anchors(
        assigned,
        anchors_per_date=expectations.anchors_per_date,
        minimum_anchors_per_date=expectations.minimum_anchors_per_date,
    )
    scoring = build_scoring_sets(assigned, anchors)

    class_counts = manifest["label"].value_counts().sort_index().astype(int).to_dict()
    day_counts = manifest.groupby("day_number").size().astype(int).to_dict()
    short_actual = {int(day): int(day_counts.get(day, 0)) for day in expectations.short_day_counts}
    regular_actual = {
        int(day): int(count)
        for day, count in day_counts.items()
        if int(day) not in set(expectations.short_day_counts)
    }
    scored_split_counts = scoring.scored["split"].value_counts().to_dict()
    for name in ("train", "validation", "test"):
        scored_split_counts.setdefault(name, 0)
    environments = assigned.loc[assigned["split"].eq("train")].groupby("environment")["date"].nunique()

    checks: dict[str, AuditCheck] = {
        "archive_size": _check(archive.size_bytes, expectations.expected_size_bytes),
        "archive_md5": _check(archive.md5, expectations.expected_md5.casefold()),
        "archive_sha256": _check(
            archive.sha256 if expectations.expected_sha256 is not None else True,
            expectations.expected_sha256.casefold() if expectations.expected_sha256 is not None else True,
        ),
        "raw_measurement_files": _check(source.raw_measurement_count, expectations.expected_measurements),
        "measurement_rows": _check(len(manifest), expectations.expected_measurements),
        "official_feature_rows": _check(
            len(feature_table),
            len(feature_table)
            if expectations.expected_official_feature_rows is None
            else expectations.expected_official_feature_rows,
            detail="Official extracted features are audited for provenance only and are not the formal sample manifest.",
        ),
        "valid_dates": _check(int(manifest["date"].nunique()), expectations.expected_valid_dates),
        "class_counts": _check(class_counts, dict(expectations.expected_class_counts)),
        "excluded_days_absent": _check(
            sorted(set(expectations.excluded_day_numbers).intersection(int(day) for day in day_counts)),
            [],
        ),
        "short_day_counts": _check(short_actual, dict(expectations.short_day_counts)),
        "regular_day_counts": _check(
            sorted(set(regular_actual.values())),
            [expectations.regular_day_count],
        ),
        "date_split": _check(
            [len(split.train_dates), len(split.validation_dates), len(split.test_dates)],
            [expectations.train_dates, expectations.validation_dates, expectations.test_dates],
        ),
        "training_environments": _check(
            {
                "count": int(environments.size),
                "dates_per_environment": sorted(environments.astype(int).unique().tolist()),
            },
            {
                "count": expectations.train_dates // expectations.dates_per_environment,
                "dates_per_environment": [expectations.dates_per_environment],
            },
        ),
        "within_day_order_complete": _check(int(manifest["within_day_order"].notna().sum()), len(manifest)),
        "blank_anchor_count": _check(
            len(scoring.anchors),
            expectations.expected_valid_dates * expectations.anchors_per_date
            if expectations.expected_blank_anchor_count is None
            else expectations.expected_blank_anchor_count,
        ),
        "common_scoring_set": _check(
            {
                budget: sorted(scoring.scored_for_budget(budget)["sample_id"].astype(str).tolist())
                == sorted(scoring.scored["sample_id"].astype(str).tolist())
                for budget in (0, 1, 3)
            },
            {0: True, 1: True, 3: True},
        ),
        "scored_split_counts": _check(
            {key: int(scored_split_counts[key]) for key in ("train", "validation", "test")},
            dict(expectations.expected_scored_split_counts),
        ),
        "sensor_feature_groups": _check(
            {
                "sensors": len(feature_groups),
                "features": sum(len(columns) for columns in feature_groups.values()),
            },
            {
                "sensors": expectations.expected_sensors,
                "features": expectations.expected_sensors * expectations.expected_features_per_sensor,
            },
        ),
    }

    all_passed = all(check.passed for check in checks.values())
    if debug:
        if not all_passed:
            failed = [name for name, check in checks.items() if not check.passed]
            raise AssertionError(f"Synthetic DEBUG audit did not pass: {failed}")
        status = "DEBUG_PASS"
    elif not archive.passed:
        status = "RETRYABLE_INFRA"
    elif all_passed:
        status = "VERIFIED_STRUCTURE"
    else:
        status = "PLAN_REVISION_REQUIRED"

    return PrimaryAuditResult(
        status=status,
        debug=debug,
        archive=archive,
        source=source,
        expectations=expectations,
        feature_table=feature_table,
        manifest=manifest,
        assigned_manifest=assigned,
        split=split,
        scoring=scoring,
        feature_groups=feature_groups,
        checks=checks,
        alignment_mode=alignment_mode,
        warnings=warnings,
    )
