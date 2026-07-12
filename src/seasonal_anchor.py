"""
src/seasonal_anchor.py

The smoothed lag-52 seasonal-naive anchor, extracted as a reusable function.

`forecast = anchor + model(residual)` is the recipe that already produced this
project's best classical result (`src/arima_forecaster.py`, Kaggle 3104/3239):
the anchor carries the strong yearly retail seasonality, and the model only has
to learn the deviation from it. Lag-52 always reaches back into the training
period for every test week (test is <=39 weeks past train's end), so the anchor
is defined over the whole horizon with no recursion.

The anchor is a 3-point average around lag-52 (anchor_window=1: t-53, t-52,
t-51) rather than the single point y_{t-52}. That setting is not a guess - it
was benchmarked on a 250-series sample of the shared holdout fold and beat the
exact single-point anchor (WMAE 1379.72 vs 1403.32), because one week's value a
year ago is noisy while the average of the 3 surrounding weeks is a steadier
estimate of "what that time of year looks like". Wider windows made things worse
(window=2 -> 1438, window=3 -> 1507): they smear in weeks whose seasonal
position is genuinely different. See the module docstring of
`src/arima_forecaster.py` for the full benchmark notes.

This logic mirrors `ArimaForecaster._prepare_series` / `._lookup`, which is
deliberately left untouched (it produced a scored submission). The two could be
unified later by pointing ArimaForecaster at this function.

No em dashes anywhere (project code-style rule) - hyphens only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smoothed_seasonal_anchor(
    history: pd.Series,
    target_dates,
    m: int = 52,
    anchor_window: int = 1,
    tolerance_days: int = 3,
    fallback_level: float | None = None,
) -> np.ndarray:
    """Seasonal-naive anchor for `target_dates`, read off `history` at lag `m`.

    Args:
        history: past values indexed by a sorted DatetimeIndex. May be shorter
            than `m` weeks, may have gaps, may be empty.
        target_dates: the dates to produce an anchor for. These may lie inside
            `history` (deseasonalizing the training period) or beyond its end
            (forecasting), since the lookup only ever reads backwards.
        m: seasonal period in weeks (52 = one year of W-FRI weeks).
        anchor_window: half-width of the averaging window around the lag-m
            point. 1 averages y_{t-53}, y_{t-52}, y_{t-51}.
        tolerance_days: how far the lookup may snap to the nearest available
            week, so small gaps in a series do not void the anchor.
        fallback_level: value used when no lag point is within tolerance (early
            weeks with no lag-m history yet, or a gap wider than tolerance).
            Defaults to the mean of `history`; if `history` is empty, 0.0.

    Returns:
        float array aligned to `target_dates`, never NaN.
    """
    target_index = pd.DatetimeIndex(target_dates)

    if fallback_level is None:
        fallback_level = float(history.mean()) if len(history) else 0.0
    if not np.isfinite(fallback_level):
        fallback_level = 0.0

    if history is None or len(history) == 0:
        return np.full(len(target_index), float(fallback_level))

    history = history.sort_index()
    lag = pd.Timedelta(weeks=m)
    tolerance = pd.Timedelta(days=tolerance_days)

    # One vectorized nearest-lookup per offset in the window, rather than a
    # per-date Python loop: the notebook calls this for ~3,300 series.
    lookups = []
    for offset in range(-anchor_window, anchor_window + 1):
        wanted = target_index - lag + pd.Timedelta(weeks=offset)
        pos = history.index.get_indexer(wanted, method="nearest", tolerance=tolerance)
        vals = np.where(pos == -1, np.nan, history.to_numpy()[pos])
        lookups.append(vals)

    stacked = np.vstack(lookups)
    found = np.isfinite(stacked)
    n_found = found.sum(axis=0)
    total = np.where(found, stacked, 0.0).sum(axis=0)

    # Dates with no lag point at all within tolerance (typically the first ~52
    # weeks of a series) fall back to the level, so the anchor is never NaN and
    # every residual series stays full-length.
    return np.where(n_found > 0, total / np.maximum(n_found, 1), float(fallback_level))
