"""Dataset discovery and parsing for the electronic-nose experiments.

The functions in this module keep acquisition metadata separate from predictor
columns.  Filenames may be used to recover supervised labels or documented
sample identities, but are never returned as response-model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Iterator, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd


UCI_CLASS_NAMES: Mapping[int, str] = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}


@dataclass(frozen=True)
class PrimarySource:
    """Resolved locations for the one-year primary dataset."""

    root: Path
    archive_path: Path | None = None
    extracted_features_path: Path | None = None
    extracted_features_member: str | None = None
    raw_csv_paths: tuple[Path, ...] = ()
    raw_csv_members: tuple[str, ...] = ()

    @property
    def has_feature_table(self) -> bool:
        return self.extracted_features_path is not None or self.extracted_features_member is not None

    @property
    def raw_measurement_count(self) -> int:
        return len(self.raw_csv_paths) + len(self.raw_csv_members)


@dataclass(frozen=True)
class RawMeasurement:
    """One raw measurement and its metadata-only source identifier."""

    sample_name: str
    frame: pd.DataFrame


def _zip_inventory(path: Path) -> tuple[str | None, tuple[str, ...]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(
                name
                for name in archive.namelist()
                if not name.endswith("/") and not name.startswith("__MACOSX/")
            )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a readable ZIP archive: {path}") from exc

    feature_members = [
        name for name in members if PurePosixPath(name).name.casefold() == "extracted_features.csv"
    ]
    feature_member = min(feature_members, key=lambda item: (item.count("/"), len(item))) if feature_members else None
    raw_members = tuple(
        sorted(
            name
            for name in members
            if name.casefold().endswith(".csv") and name != feature_member
        )
    )
    return feature_member, raw_members


def discover_primary_source(source: str | Path) -> PrimarySource:
    """Discover an extracted directory, feature CSV, or intact Dataset.zip.

    A partially downloaded ``*.part`` file is deliberately ignored.  ZIP files
    are read in place so a full local extraction is not required for audit or
    minimal-sample testing.
    """

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.suffix.casefold() == ".zip":
            feature_member, raw_members = _zip_inventory(path)
            return PrimarySource(
                root=path.parent,
                archive_path=path,
                extracted_features_member=feature_member,
                raw_csv_members=raw_members,
            )
        if path.name.casefold() == "extracted_features.csv":
            return PrimarySource(root=path.parent, extracted_features_path=path)
        raise ValueError(f"Expected a directory, Dataset.zip, or extracted_features.csv: {path}")

    files = [item for item in path.rglob("*") if item.is_file()]
    feature_paths = [item for item in files if item.name.casefold() == "extracted_features.csv"]
    feature_path = min(feature_paths, key=lambda item: (len(item.relative_to(path).parts), len(str(item)))) if feature_paths else None
    raw_paths = tuple(
        sorted(
            (item for item in files if item.suffix.casefold() == ".csv" and item != feature_path),
            key=lambda item: str(item).casefold(),
        )
    )

    zip_paths = [
        item
        for item in files
        if item.suffix.casefold() == ".zip" and not item.name.casefold().endswith(".zip.part")
    ]
    archive_path: Path | None = None
    feature_member: str | None = None
    raw_members: tuple[str, ...] = ()
    if not feature_path and zip_paths:
        archive_path = min(
            zip_paths,
            key=lambda item: (item.name.casefold() != "dataset.zip", len(item.relative_to(path).parts), len(str(item))),
        )
        feature_member, raw_members = _zip_inventory(archive_path)

    if feature_path is None and archive_path is None and not raw_paths:
        raise FileNotFoundError(f"No complete primary-dataset files found under {path}")

    return PrimarySource(
        root=path,
        archive_path=archive_path,
        extracted_features_path=feature_path,
        extracted_features_member=feature_member,
        raw_csv_paths=raw_paths,
        raw_csv_members=raw_members,
    )


def _decode_table_bytes(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", payload, 0, 1, "unable to decode table")


def _infer_separator(text: str) -> str:
    first = next((line for line in text.splitlines() if line.strip()), "")
    counts = {",": first.count(","), ";": first.count(";"), "\t": first.count("\t")}
    delimiter, count = max(counts.items(), key=lambda pair: pair[1])
    return delimiter if count > 0 else r"\s+"


def read_delimited_auto(source: str | Path | bytes, **kwargs: object) -> pd.DataFrame:
    """Read comma, semicolon, tab, or whitespace separated text deterministically."""

    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    text = _decode_table_bytes(payload)
    separator = kwargs.pop("sep", _infer_separator(text))
    if separator == r"\s+":
        nrows = kwargs.pop("nrows", None)
        if kwargs:
            raise TypeError(f"Unsupported quoted-whitespace reader options: {sorted(kwargs)}")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Whitespace-delimited table is empty")
        header = shlex.split(lines[0], posix=True)
        data_lines = lines[1:] if nrows is None else lines[1 : 1 + int(nrows)]
        rows = [shlex.split(line, posix=True) for line in data_lines]
        invalid = [index + 2 for index, row in enumerate(rows) if len(row) != len(header)]
        if invalid:
            raise ValueError(
                f"Quoted-whitespace table has inconsistent field counts at lines {invalid[:5]}"
            )
        frame = pd.DataFrame(rows, columns=header)
        for column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() == frame[column].astype(str).str.strip().ne("").sum():
                frame[column] = converted
        return frame
    return pd.read_csv(BytesIO(text.encode("utf-8")), sep=separator, engine="python", **kwargs)


def load_primary_features(source: str | Path | PrimarySource) -> pd.DataFrame:
    """Load the official feature table from a directory, CSV, or ZIP archive."""

    resolved = source if isinstance(source, PrimarySource) else discover_primary_source(source)
    if resolved.extracted_features_path is not None:
        frame = read_delimited_auto(resolved.extracted_features_path)
        frame.attrs["source"] = str(resolved.extracted_features_path)
        return frame
    if resolved.archive_path is not None and resolved.extracted_features_member is not None:
        with zipfile.ZipFile(resolved.archive_path) as archive:
            frame = read_delimited_auto(archive.read(resolved.extracted_features_member))
        frame.attrs["source"] = f"{resolved.archive_path}!/{resolved.extracted_features_member}"
        return frame
    raise FileNotFoundError("extracted_features.csv was not found in the resolved source")


def iter_primary_raw_measurements(
    source: str | Path | PrimarySource,
    *,
    limit: int | None = None,
) -> Iterator[RawMeasurement]:
    """Yield raw measurement tables without extracting a complete archive."""

    resolved = source if isinstance(source, PrimarySource) else discover_primary_source(source)
    emitted = 0
    for path in resolved.raw_csv_paths:
        if limit is not None and emitted >= limit:
            return
        yield RawMeasurement(sample_name=str(path.relative_to(resolved.root)), frame=read_delimited_auto(path))
        emitted += 1

    if resolved.archive_path is not None:
        with zipfile.ZipFile(resolved.archive_path) as archive:
            for member in resolved.raw_csv_members:
                if limit is not None and emitted >= limit:
                    return
                yield RawMeasurement(sample_name=member, frame=read_delimited_auto(archive.read(member)))
                emitted += 1


def _normalise_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {_normalise_column_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(_normalise_column_name(candidate))
        if match is not None:
            return match
    return None


def parse_primary_label(value: object, numeric_label_map: Mapping[int, str] | None = None) -> str:
    """Return a canonical operational-class label.

    Numeric labels require an explicit mapping because their meaning must not be
    guessed from row order or an undocumented encoding.
    """

    if pd.isna(value):
        raise ValueError("Missing primary-dataset class label")
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()):
        key = int(value)
        if numeric_label_map is None or key not in numeric_label_map:
            raise ValueError(f"Numeric class label {key} requires an explicit numeric_label_map")
        return str(numeric_label_map[key])

    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        key = int(float(text))
        if numeric_label_map is None or key not in numeric_label_map:
            raise ValueError(f"Numeric class label {key} requires an explicit numeric_label_map")
        return str(numeric_label_map[key])

    folded = re.sub(r"[_\-]+", " ", text.casefold())
    # Match the longer analyte name first: "phenylethanol" contains the
    # substring "ethanol" and must never be collapsed into the EtOH class.
    if "phenyl" in folded or "phenylethanol" in folded:
        if re.search(r"(?<!\d)1000(?!\d)", folded):
            return "2-Phenylethanol 1000 ppm"
        if re.search(r"(?<!\d)200(?!\d)", folded):
            return "2-Phenylethanol 200 ppm"
        return "2-Phenylethanol"
    if "etoh" in folded or "ethanol" in folded:
        return "EtOH"
    if "diacetyl" in folded:
        if re.search(r"(?<!\d)0[.,]1(?!\d)", folded):
            return "Diacetyl 0.1 ppm"
        if re.search(r"(?<!\d)1(?:[.,]0+)?(?!\d)", folded):
            return "Diacetyl 1 ppm"
        return "Diacetyl"
    if not text:
        raise ValueError("Empty primary-dataset class label")
    return text


def _extract_day_number(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()):
        return int(value)
    match = re.search(r"(?:^|[^a-z])day[\s_-]*0*(\d+)(?:[^0-9]|$)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_ethanol_replicate(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(?:etoh|ethanol)[^0-9]{0,12}([1-6])(?:[^0-9]|$)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _time_rank(values: pd.Series, groups: pd.Series) -> pd.Series:
    parsed = pd.to_timedelta(values.astype(str), errors="coerce").dt.total_seconds()
    if parsed.isna().all():
        parsed = pd.to_datetime(values, errors="coerce").astype("int64").astype(float)
        parsed[parsed < 0] = np.nan
    return parsed.groupby(groups).rank(method="first").astype("Int64")


def build_primary_manifest(
    feature_table: pd.DataFrame,
    *,
    source_names: Sequence[str] | None = None,
    numeric_label_map: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Build a metadata-only sample manifest from the official feature table.

    The returned ``source_name`` and order fields are audit metadata.  Response
    model code must select predictor columns separately and must not use any
    manifest column as an input feature.
    """

    frame = feature_table.reset_index(drop=True)
    if source_names is not None and len(source_names) != len(frame):
        raise ValueError("source_names length does not match the feature table")

    label_col = _find_column(frame, ("Class", "Label", "Gas", "Analyte"))
    date_col = _find_column(frame, ("Date[yyyy-mm-dd]", "Date", "MeasurementDate", "AcquisitionDate"))
    day_col = _find_column(frame, ("Day", "DayID", "MeasurementDay"))
    name_col = _find_column(frame, ("Filename", "FileName", "File", "Path", "SampleName", "SampleID"))
    order_col = _find_column(frame, ("WithinDayOrder", "Position", "MeasurementOrder", "Sequence"))
    time_col = _find_column(frame, ("Time[hh:mm:ss]", "Time", "StartTime", "Timestamp"))

    if source_names is not None:
        names = pd.Series(list(source_names), dtype="string")
    elif name_col is not None:
        names = frame[name_col].astype("string")
    else:
        unnamed = next((column for column in frame.columns if str(column).casefold().startswith("unnamed")), None)
        names = frame[unnamed].astype("string") if unnamed is not None else pd.Series(pd.NA, index=frame.index, dtype="string")

    day_numbers = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if day_col is not None:
        day_numbers = frame[day_col].map(_extract_day_number).astype("Int64")
    missing_day = day_numbers.isna()
    day_numbers.loc[missing_day] = names.loc[missing_day].map(_extract_day_number).astype("Int64")

    if date_col is not None:
        parsed_dates = pd.to_datetime(frame[date_col], errors="coerce")
        date_keys = parsed_dates.dt.strftime("%Y-%m-%d").astype("string")
    else:
        date_keys = pd.Series(pd.NA, index=frame.index, dtype="string")
    missing_date = date_keys.isna()
    date_keys.loc[missing_date] = day_numbers.loc[missing_date].map(
        lambda value: f"Day {int(value):03d}" if not pd.isna(value) else pd.NA
    )
    if date_keys.isna().any():
        raise ValueError("Could not determine an actual date or documented Day identifier for every sample")

    if label_col is not None:
        raw_labels = frame[label_col]
    elif names.notna().all():
        raw_labels = names
    else:
        raise ValueError("No Class/Label column or source filename is available for supervised targets")
    labels = raw_labels.map(lambda value: parse_primary_label(value, numeric_label_map))

    if order_col is not None:
        within_order = pd.to_numeric(frame[order_col], errors="coerce").astype("Int64")
    elif time_col is not None:
        within_order = _time_rank(frame[time_col], date_keys)
    else:
        within_order = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    sample_ids = names.copy()
    generated = pd.Series([f"sample_{index:06d}" for index in range(len(frame))], dtype="string")
    sample_ids = sample_ids.where(sample_ids.notna() & sample_ids.str.strip().ne(""), generated)
    if sample_ids.duplicated().any():
        duplicate_rank = sample_ids.groupby(sample_ids, sort=False).cumcount()
        sample_ids = sample_ids.astype(str) + duplicate_rank.map(lambda value: "" if value == 0 else f"#{value}")

    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "date": date_keys.astype(str),
            "day_number": day_numbers,
            "label": labels.astype(str),
            "within_day_order": within_order,
            "ethanol_replicate": names.map(_extract_ethanol_replicate).astype("Int64"),
            "source_name": names,
        }
    )
    manifest.attrs["predictor_warning"] = "Manifest fields are metadata and must not be used by response models."
    return manifest


def _read_uci_lines(
    payload: bytes,
    *,
    batch: int,
    expected_features: int,
    require_dense: bool,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    text = _decode_table_bytes(payload)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        header = tokens.pop(0)
        if ";" not in header:
            raise ValueError(f"Batch {batch}, line {line_number}: expected 'class;concentration'")
        class_text, concentration_text = header.split(";", maxsplit=1)
        class_id = int(class_text)
        if class_id not in UCI_CLASS_NAMES:
            raise ValueError(f"Batch {batch}, line {line_number}: unknown class {class_id}")
        feature_values: dict[int, float] = {}
        for token in tokens:
            if ":" not in token:
                raise ValueError(f"Batch {batch}, line {line_number}: invalid feature token {token!r}")
            index_text, value_text = token.split(":", maxsplit=1)
            index = int(index_text)
            if index < 1 or index > expected_features or index in feature_values:
                raise ValueError(f"Batch {batch}, line {line_number}: invalid feature index {index}")
            feature_values[index] = float(value_text)
        if require_dense and len(feature_values) != expected_features:
            missing = sorted(set(range(1, expected_features + 1)).difference(feature_values))
            raise ValueError(f"Batch {batch}, line {line_number}: missing feature indices {missing[:8]}")

        row: dict[str, float | int | str] = {
            "sample_id": f"batch{batch:02d}_{len(rows):06d}",
            "batch": batch,
            "class_id": class_id,
            "label": UCI_CLASS_NAMES[class_id],
            "concentration": float(concentration_text),
        }
        for index in range(1, expected_features + 1):
            row[f"f{index:03d}"] = feature_values.get(index, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def read_uci_libsvm_file(
    source: str | Path | bytes,
    *,
    batch: int,
    expected_features: int = 128,
    require_dense: bool = True,
) -> pd.DataFrame:
    """Read one UCI ``batch*.dat`` file with ``class;concentration x:v`` rows."""

    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    return _read_uci_lines(
        payload,
        batch=batch,
        expected_features=expected_features,
        require_dense=require_dense,
    )


def read_uci_batches(
    source: str | Path,
    *,
    expected_features: int = 128,
    require_batches: Sequence[int] | None = tuple(range(1, 11)),
    expected_measurements: int | None = None,
    require_exact_batches: bool = True,
) -> pd.DataFrame:
    """Read chronologically numbered UCI batch files from a directory or ZIP."""

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    payloads: list[tuple[int, str, bytes]] = []
    if path.is_file() and path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                match = re.search(r"batch\s*0*(\d+)\.dat$", PurePosixPath(member).name, flags=re.IGNORECASE)
                if match:
                    payloads.append((int(match.group(1)), member, archive.read(member)))
    elif path.is_dir():
        for item in path.rglob("*.dat"):
            match = re.search(r"batch\s*0*(\d+)\.dat$", item.name, flags=re.IGNORECASE)
            if match:
                payloads.append((int(match.group(1)), str(item), item.read_bytes()))
    else:
        raise ValueError(f"Expected a directory or ZIP containing batch*.dat: {path}")

    if not payloads:
        raise FileNotFoundError(f"No batch*.dat files found in {path}")
    seen: dict[int, str] = {}
    for batch, name, _ in payloads:
        if batch in seen:
            raise ValueError(f"Duplicate UCI batch {batch}: {seen[batch]} and {name}")
        seen[batch] = name
    if require_batches is not None:
        required_set = set(require_batches)
        missing = sorted(required_set.difference(seen))
        if missing:
            raise ValueError(f"Missing required UCI batches: {missing}")
        if require_exact_batches:
            unexpected = sorted(set(seen).difference(required_set))
            if unexpected:
                raise ValueError(f"Unexpected UCI batches: {unexpected}")

    frames = [
        _read_uci_lines(payload, batch=batch, expected_features=expected_features, require_dense=True)
        for batch, _, payload in sorted(payloads, key=lambda item: item[0])
    ]
    result = pd.concat(frames, ignore_index=True)
    if expected_measurements is not None and len(result) != int(expected_measurements):
        raise ValueError(
            f"UCI identity gate expected {int(expected_measurements)} measurements, found {len(result)}."
        )
    result.attrs["predictor_columns"] = [f"f{index:03d}" for index in range(1, expected_features + 1)]
    result.attrs["metadata_columns"] = ["sample_id", "batch", "class_id", "label", "concentration"]
    result.attrs["identity_audit"] = {
        "measurements": len(result),
        "expected_measurements": None if expected_measurements is None else int(expected_measurements),
        "features": int(expected_features),
        "batches": sorted(int(value) for value in seen),
        "exact_batches_required": bool(require_exact_batches),
    }
    return result


def uci_sensor_feature_groups(
    *,
    n_sensors: int = 16,
    features_per_sensor: int = 8,
) -> dict[int, tuple[str, ...]]:
    """Return the documented sensor-major grouping of UCI feature columns."""

    return {
        sensor: tuple(
            f"f{((sensor - 1) * features_per_sensor + offset):03d}"
            for offset in range(1, features_per_sensor + 1)
        )
        for sensor in range(1, n_sensors + 1)
    }
