"""
src/validation.py

Time-based cross-validation splitter, shared by both model tracks (tree/
classical and darts-based DL) so WMAE numbers are comparable across all
architectures. Never use random splits on this data - a random split
leaks future weeks into training.

Splits are expressed purely as date boundaries (Split), not row indices,
so the same fold definitions apply whether the caller holds a flat
DataFrame (Store, Dept, Date, ...) or a list of per-series darts
TimeSeries. Two adapters are provided: split_frame() for the flat-frame
case and split_series() for a single darts TimeSeries.

"""

from dataclasses import dataclass

import pandas as pd

from src.features import HOLIDAY_DATES

TEST_HORIZON_WEEKS = 39
DEFAULT_VAL_WEEKS = 13


@dataclass(frozen=True)
class Split:
    """
    One CV fold, expressed as date boundaries (inclusive on both ends).

    train_start=None means "from the start of available history"
    (expanding window). Set explicitly for a sliding/fixed-size window.
    """

    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_start: pd.Timestamp | None = None


def _unique_sorted_dates(dates) -> list[pd.Timestamp]:
    """Coerce a date-like array/Series/Index into a sorted list of unique Timestamps."""
    return [pd.Timestamp(d) for d in sorted(pd.to_datetime(dates).unique())]


def expanding_window_splits(
    dates,
    n_splits: int,
    val_weeks: int = DEFAULT_VAL_WEEKS,
    step_weeks: int | None = None,
    min_train_weeks: int = 52,
) -> list[Split]:
    """
    Expanding-window CV: every fold trains on everything from the start
    of history up to train_end, and validates on the val_weeks after it.

    Stops early (returns fewer than n_splits folds) if there isn't
    enough history left to satisfy min_train_weeks - check len() of the
    result rather than assuming you get exactly n_splits back.
    """
    step_weeks = step_weeks or val_weeks
    unique_dates = _unique_sorted_dates(dates)
    total = len(unique_dates)

    splits = []
    val_end_idx = total - 1
    for _ in range(n_splits):
        val_start_idx = val_end_idx - val_weeks + 1
        train_end_idx = val_start_idx - 1
        if val_start_idx < 0 or train_end_idx < min_train_weeks - 1:
            break
        splits.append(
            Split(
                train_end=unique_dates[train_end_idx],
                val_start=unique_dates[val_start_idx],
                val_end=unique_dates[val_end_idx],
            )
        )
        val_end_idx -= step_weeks

    splits.reverse()  # oldest fold first, chronological order
    return splits


def sliding_window_splits(
    dates,
    n_splits: int,
    val_weeks: int = DEFAULT_VAL_WEEKS,
    train_weeks: int = 52,
    step_weeks: int | None = None,
) -> list[Split]:
    """
    Fixed-size training window CV: every fold trains on exactly
    train_weeks of history immediately before val_start, instead of
    everything since the start of history. Useful where re-fitting on
    the full expanding history is expensive (e.g. re-tuned ARIMA per
    fold) or where old history is suspected to be less relevant.

    Same backward-anchored construction as expanding_window_splits, and
    the same "stops early if history runs out" behavior.
    """
    step_weeks = step_weeks or val_weeks
    unique_dates = _unique_sorted_dates(dates)
    total = len(unique_dates)

    splits = []
    val_end_idx = total - 1
    for _ in range(n_splits):
        val_start_idx = val_end_idx - val_weeks + 1
        train_end_idx = val_start_idx - 1
        train_start_idx = train_end_idx - train_weeks + 1
        if val_start_idx < 0 or train_start_idx < 0:
            break
        splits.append(
            Split(
                train_end=unique_dates[train_end_idx],
                val_start=unique_dates[val_start_idx],
                val_end=unique_dates[val_end_idx],
                train_start=unique_dates[train_start_idx],
            )
        )
        val_end_idx -= step_weeks

    splits.reverse()
    return splits


def split_frame(
    df: pd.DataFrame, split: Split, date_col: str = "Date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a Split to a flat (Store, Dept, Date, ...) DataFrame. Returns (train_df, val_df)."""
    dates = pd.to_datetime(df[date_col])
    train_mask = dates <= split.train_end
    if split.train_start is not None:
        train_mask &= dates >= split.train_start
    val_mask = (dates >= split.val_start) & (dates <= split.val_end)
    return df.loc[train_mask].copy(), df.loc[val_mask].copy()


def split_series(series, split: Split):
    """
    Apply a Split to a single darts TimeSeries. Returns (train_series,
    val_series), using TimeSeries.slice(start, end).
    """
    train_start = split.train_start if split.train_start is not None else series.start_time()
    train_series = series.slice(train_start, split.train_end)
    val_series = series.slice(split.val_start, split.val_end)
    return train_series, val_series


def holiday_dates_flat() -> list[pd.Timestamp]:
    """Flatten features.HOLIDAY_DATES into one sorted list, for fold coverage checks."""
    flat = [d for holiday_map in HOLIDAY_DATES.values() for d in holiday_map.values()]
    return sorted(pd.to_datetime(flat))


def describe_split(split: Split, holiday_dates: list[pd.Timestamp] | None = None) -> dict:
    """
    Summarize a Split as a flat dict - handy for mlflow.log_params (CV
    strategy) and for sanity-checking holiday coverage per fold, since
    WMAE weights holiday weeks 5x and there are only 4 per year in the
    data (a fold with zero holidays gives a less trustworthy estimate).
    """
    holiday_dates = holiday_dates if holiday_dates is not None else holiday_dates_flat()
    n_holidays_in_val = sum(split.val_start <= d <= split.val_end for d in holiday_dates)
    return {
        "train_start": str(split.train_start) if split.train_start is not None else "start_of_history",
        "train_end": str(split.train_end),
        "val_start": str(split.val_start),
        "val_end": str(split.val_end),
        "val_weeks": (split.val_end - split.val_start).days // 7 + 1,
        "n_holidays_in_val": n_holidays_in_val,
    }
