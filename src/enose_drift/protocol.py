"""Time-causal dataset protocols and blank-anchor identities."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrimaryDateSplit:
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]

    @property
    def all_dates(self) -> tuple[str, ...]:
        return self.train_dates + self.validation_dates + self.test_dates

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train_dates),
            "validation": list(self.validation_dates),
            "test": list(self.test_dates),
        }


@dataclass(frozen=True)
class ScoringSets:
    """Common scored observations and candidate anchors for every B budget."""

    all_samples: pd.DataFrame
    anchors: pd.DataFrame
    scored: pd.DataFrame

    def anchors_for_budget(self, budget: int, *, allow_partial: bool = False) -> pd.DataFrame:
        return select_anchors_for_budget(self.anchors, budget, allow_partial=allow_partial)

    def scored_for_budget(self, budget: int) -> pd.DataFrame:
        if budget not in (0, 1, 3):
            raise ValueError("Blank budget must be one of 0, 1, or 3")
        return self.scored.copy()


@dataclass(frozen=True)
class TemporalFrameSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ExpandingWindowFold:
    test_key: object
    history_keys: tuple[object, ...]
    train_keys: tuple[object, ...]
    validation_keys: tuple[object, ...]
    history: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _chronological_keys(values: pd.Series) -> list[str]:
    keys = values.dropna().astype(str).drop_duplicates().tolist()
    if not keys:
        raise ValueError("No chronological keys were found")

    parsed_dates = pd.to_datetime(pd.Series(keys), errors="coerce")
    if parsed_dates.notna().all():
        return [key for _, key in sorted(zip(parsed_dates.tolist(), keys), key=lambda pair: pair[0])]

    day_numbers: list[tuple[int, str]] = []
    for key in keys:
        match = re.fullmatch(r"\s*Day\s*0*(\d+)\s*", key, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Chronological key is neither an actual date nor 'Day N': {key!r}")
        day_numbers.append((int(match.group(1)), key))
    return [key for _, key in sorted(day_numbers, key=lambda pair: pair[0])]


def chronological_date_split(
    manifest: pd.DataFrame,
    *,
    date_col: str = "date",
    n_train: int = 24,
    n_validation: int = 7,
    n_test: int = 8,
    require_exact: bool = True,
) -> PrimaryDateSplit:
    """Create a fixed forward split while keeping every date intact."""

    if date_col not in manifest:
        raise KeyError(date_col)
    dates = _chronological_keys(manifest[date_col])
    required = n_train + n_validation + n_test
    if require_exact and len(dates) != required:
        raise ValueError(f"Expected exactly {required} dates, found {len(dates)}")
    if len(dates) < required:
        raise ValueError(f"Need at least {required} dates, found {len(dates)}")
    if len(dates) > required:
        raise ValueError("Extra dates require an explicit plan revision; the split boundary is frozen")

    split = PrimaryDateSplit(
        train_dates=tuple(dates[:n_train]),
        validation_dates=tuple(dates[n_train : n_train + n_validation]),
        test_dates=tuple(dates[n_train + n_validation :]),
    )
    if len(set(split.all_dates)) != len(split.all_dates):
        raise AssertionError("A date was assigned to more than one partition")
    return split


def assign_primary_split(
    manifest: pd.DataFrame,
    split: PrimaryDateSplit,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add a metadata-only partition column to a manifest."""

    mapping = {date: "train" for date in split.train_dates}
    mapping.update({date: "validation" for date in split.validation_dates})
    mapping.update({date: "test" for date in split.test_dates})
    result = manifest.copy()
    result["split"] = result[date_col].astype(str).map(mapping)
    if result["split"].isna().any():
        unknown = sorted(result.loc[result["split"].isna(), date_col].astype(str).unique())
        raise ValueError(f"Manifest contains dates outside the frozen split: {unknown}")
    if result.groupby(date_col)["split"].nunique().max() != 1:
        raise AssertionError("At least one date was split across partitions")
    return result


def build_training_environments(
    assigned_manifest: pd.DataFrame,
    *,
    date_col: str = "date",
    split_col: str = "split",
    dates_per_environment: int = 4,
) -> pd.DataFrame:
    """Assign the 24 training dates to consecutive four-date environments."""

    if dates_per_environment <= 0:
        raise ValueError("dates_per_environment must be positive")
    result = assigned_manifest.copy()
    train_dates = _chronological_keys(result.loc[result[split_col].eq("train"), date_col])
    if len(train_dates) % dates_per_environment:
        raise ValueError("Training dates do not divide evenly into complete temporal environments")
    date_to_environment = {
        date: index // dates_per_environment
        for index, date in enumerate(train_dates)
    }
    result["environment"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    train_mask = result[split_col].eq("train")
    result.loc[train_mask, "environment"] = (
        result.loc[train_mask, date_col].astype(str).map(date_to_environment).astype("Int64")
    )
    return result


def _is_etoh(value: object) -> bool:
    folded = str(value).strip().casefold()
    return folded in {"etoh", "ethanol"}


def identify_blank_anchors(
    manifest: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    date_col: str = "date",
    label_col: str = "label",
    order_col: str = "within_day_order",
    ethanol_replicate_col: str = "ethanol_replicate",
    anchors_per_date: int = 3,
    minimum_anchors_per_date: int | None = None,
) -> pd.DataFrame:
    """Identify the initial contiguous EtOH references observed on each date."""

    required = {sample_id_col, date_col, label_col}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise KeyError(f"Missing manifest columns: {missing}")
    if manifest[sample_id_col].duplicated().any():
        raise ValueError("sample_id must be unique before anchor assignment")

    minimum = anchors_per_date if minimum_anchors_per_date is None else int(minimum_anchors_per_date)
    if not 1 <= minimum <= anchors_per_date:
        raise ValueError("minimum_anchors_per_date must lie between one and anchors_per_date")
    anchors: list[pd.DataFrame] = []
    for date, date_frame in manifest.groupby(date_col, sort=False):
        if order_col not in date_frame or date_frame[order_col].isna().any():
            raise ValueError(f"Date {date!r} lacks complete observed within-day order")
        ordered = date_frame.sort_values(order_col, kind="stable")
        initial_indices: list[object] = []
        for index, row in ordered.iterrows():
            if not _is_etoh(row[label_col]):
                break
            initial_indices.append(index)
        if len(initial_indices) < minimum:
            raise ValueError(
                f"Date {date!r} contains only {len(initial_indices)} initial causal EtOH references; "
                f"at least {minimum} are required"
            )
        selected = ordered.loc[initial_indices[:anchors_per_date]].copy()
        selected["anchor_rank"] = np.arange(1, len(selected) + 1)
        selected["protocol_role"] = "blank_anchor_candidate"
        anchors.append(selected)

    result = pd.concat(anchors, ignore_index=True)
    result["anchor_rank"] = result["anchor_rank"].astype(int)
    return result.sort_values([date_col, "anchor_rank"], kind="stable").reset_index(drop=True)


def select_anchors_for_budget(
    anchors: pd.DataFrame,
    budget: int,
    *,
    date_col: str = "date",
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Select zero, the first, or all three preceding anchors per date."""

    if budget not in (0, 1, 3):
        raise ValueError("Blank budget must be one of 0, 1, or 3")
    if budget == 0:
        return anchors.iloc[0:0].copy()
    selected = anchors.loc[anchors["anchor_rank"].le(budget)].copy()
    counts = selected.groupby(date_col).size()
    if not allow_partial and not counts.eq(budget).all():
        raise ValueError(f"Every date must provide exactly {budget} allowed anchors")
    if allow_partial and (counts.lt(1).any() or counts.gt(budget).any()):
        raise ValueError(f"Every date must provide between one and {budget} allowed anchors")
    return selected


def build_scoring_sets(
    manifest: pd.DataFrame,
    anchors: pd.DataFrame | None = None,
    *,
    sample_id_col: str = "sample_id",
) -> ScoringSets:
    """Remove all candidate anchors from one scoring set shared by B=0/1/3."""

    resolved_anchors = identify_blank_anchors(manifest) if anchors is None else anchors.copy()
    anchor_ids = set(resolved_anchors[sample_id_col].astype(str))
    if not anchor_ids.issubset(set(manifest[sample_id_col].astype(str))):
        raise ValueError("Anchor table contains sample IDs outside the manifest")

    all_samples = manifest.copy()
    all_samples["is_blank_anchor"] = all_samples[sample_id_col].astype(str).isin(anchor_ids)
    all_samples["protocol_role"] = np.where(
        all_samples["is_blank_anchor"], "blank_anchor_candidate", "scored"
    )
    scored = all_samples.loc[~all_samples["is_blank_anchor"]].copy().reset_index(drop=True)
    return ScoringSets(
        all_samples=all_samples.reset_index(drop=True),
        anchors=resolved_anchors.reset_index(drop=True),
        scored=scored,
    )


def uci_fixed_origin_splits(
    frame: pd.DataFrame,
    *,
    batch_col: str = "batch",
    train_batch: int = 1,
    validation_batch: int = 2,
    test_batches: Sequence[int] = tuple(range(3, 11)),
) -> TemporalFrameSplit:
    """Return the frozen UCI Batch 1 / Batch 2 / Batch 3--10 protocol."""

    available = set(pd.to_numeric(frame[batch_col], errors="raise").astype(int).unique())
    required = {train_batch, validation_batch, *test_batches}
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"Missing UCI batches: {missing}")
    train = frame.loc[frame[batch_col].eq(train_batch)].copy()
    validation = frame.loc[frame[batch_col].eq(validation_batch)].copy()
    test = frame.loc[frame[batch_col].isin(test_batches)].copy()
    return TemporalFrameSplit(train=train, validation=validation, test=test)


def uci_expanding_window_splits(
    frame: pd.DataFrame,
    *,
    batch_col: str = "batch",
    first_test_batch: int = 3,
    last_test_batch: int = 10,
) -> list[ExpandingWindowFold]:
    """Build UCI forward folds without exposing the current or future batch.

    ``history`` contains all labeled past batches available at an origin.  The
    latest historical batch is also exposed as ``validation`` and earlier
    batches as ``train`` for leakage-free tuning; callers may refit on the full
    history only after tuning choices are fixed.
    """

    available = set(pd.to_numeric(frame[batch_col], errors="raise").astype(int).unique())
    folds: list[ExpandingWindowFold] = []
    for test_batch in range(first_test_batch, last_test_batch + 1):
        history_keys = tuple(range(1, test_batch))
        required = {*history_keys, test_batch}
        missing = sorted(required.difference(available))
        if missing:
            raise ValueError(f"Fold ending at Batch {test_batch} is missing batches {missing}")
        validation_keys = (history_keys[-1],)
        train_keys = history_keys[:-1]
        history = frame.loc[frame[batch_col].isin(history_keys)].copy()
        train = frame.loc[frame[batch_col].isin(train_keys)].copy()
        validation = frame.loc[frame[batch_col].isin(validation_keys)].copy()
        test = frame.loc[frame[batch_col].eq(test_batch)].copy()
        folds.append(
            ExpandingWindowFold(
                test_key=test_batch,
                history_keys=history_keys,
                train_keys=train_keys,
                validation_keys=validation_keys,
                history=history,
                train=train,
                validation=validation,
                test=test,
            )
        )
    return folds


def primary_expanding_window_splits(
    manifest: pd.DataFrame,
    *,
    date_col: str = "date",
    first_test_index: int,
    n_folds: int,
) -> list[ExpandingWindowFold]:
    """Build supplemental primary-data folds using only earlier complete dates."""

    dates = _chronological_keys(manifest[date_col])
    if first_test_index < 2:
        raise ValueError("At least two historical dates are needed for train/validation separation")
    if first_test_index + n_folds > len(dates):
        raise ValueError("Requested expanding-window folds exceed the available dates")
    folds: list[ExpandingWindowFold] = []
    for index in range(first_test_index, first_test_index + n_folds):
        history_keys = tuple(dates[:index])
        validation_keys = (history_keys[-1],)
        train_keys = history_keys[:-1]
        test_key = dates[index]
        folds.append(
            ExpandingWindowFold(
                test_key=test_key,
                history_keys=history_keys,
                train_keys=train_keys,
                validation_keys=validation_keys,
                history=manifest.loc[manifest[date_col].isin(history_keys)].copy(),
                train=manifest.loc[manifest[date_col].isin(train_keys)].copy(),
                validation=manifest.loc[manifest[date_col].isin(validation_keys)].copy(),
                test=manifest.loc[manifest[date_col].eq(test_key)].copy(),
            )
        )
    return folds
