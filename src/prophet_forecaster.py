"""
src/prophet_forecaster.py

Per-(Store, Dept) Prophet forecaster for the Walmart submission.

WHY THIS IS NOT THE STORE-LEVEL DESIGN IT USED TO BE
----------------------------------------------------
The previous revision fit one Prophet model per STORE (45 series) and
recovered (Store, Dept) rows by scaling each store's forecast by that
department's constant historical share of the store's total. That design
cannot represent department-level seasonality at all: inside a store, every
department got the same seasonal curve, only rescaled. Departments do not
behave that way - electronics roughly triples in the pre-Christmas week
while grocery barely moves, and lawn/garden peaks in spring - so the only
department-specific time variation the model could express was a single
on/off holiday multiplier.

The Kaggle test horizon (2012-11-02 to 2013-07-26) is made almost entirely
of what that design could not represent: Thanksgiving and Christmas (which
WMAE weights 5x) plus a full spring season. It scored ~4,700 WMAE on the
leaderboard while its Aug-Oct holdout - which contains neither a holiday nor
a spring - reported ~2,900. The holdout was not measuring the failure mode.

This revision fits one Prophet model per (Store, Dept) series (~3,300 of
them), which is the entire point: seasonality and holiday coefficients are
now learned per department. It is affordable because a MAP fit on ~143
weekly points with uncertainty sampling switched off takes a fraction of a
second, and the fits are embarrassingly parallel.

Series too short for a yearly seasonality to be identifiable, and (Store,
Dept) pairs that appear in test but never in train, fall back to a
department-level national seasonal profile scaled to that series' own level
- see `FallbackStats`. Even the fallback path therefore carries department
seasonality, which the old constant-share design never did.

Prophet models are serialized with prophet.serialize.model_to_json rather
than raw pickle - that round-trips through joblib.dump and MLflow artifact
logging far more reliably than a cmdstanpy/Stan-backed object would. Dump
the pipeline with compression (joblib.dump(..., compress=3)); ~3,300 model
JSON strings are a few hundred MB raw and compress to a small fraction of
that.

Worker functions are module-level so joblib's loky backend can pickle and
dispatch them - the same reasoning src/arima_forecaster.py documents for its
own per-series worker.

No em dashes anywhere (project code-style rule) - hyphens only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# A (Store, Dept) series with less than a full year of history cannot
# identify a yearly seasonality, and fitting one anyway would extrapolate
# noise across the 39-week test horizon. Those series use FallbackStats.
MIN_OBS_FOR_PROPHET = 52


# ---------------------------------------------------------------------------
# Christmas-week overlap regressors
# ---------------------------------------------------------------------------
# The dataset's weeks end on a Friday and cover the 7 days [ds-6, ds]. Where
# Christmas falls inside that grid moves from year to year, so the number of
# pre-Christmas shopping days captured by the week the competition MARKS as
# the Christmas holiday is NOT constant:
#
#   week ending 2010-12-31 covers Dec 25-31 -> 0 days of the Dec 15-24 rush
#   week ending 2011-12-30 covers Dec 24-30 -> 1 day  (Dec 24)
#   week ending 2012-12-28 covers Dec 22-28 -> 3 days (Dec 22, 23, 24)
#
# Test's Christmas week is 2012-12-28. It therefore contains three times the
# pre-Christmas trade of any Christmas week a model can learn from, and a
# plain on/off holiday dummy - which is all a `holidays` dataframe can
# express - necessarily underpredicts it. WMAE weights that week 5x and it is
# the largest sales week of the year, so this is expensive.
#
# These two regressors count the relevant December days in each week
# directly. The fitted coefficient is then per-DAY rather than per-week, so it
# extrapolates to 2012 and 2013 correctly instead of assuming every Christmas
# week looks like the training ones.
#
# Note the week-51 peak weeks (2010-12-24, 2011-12-23, 2012-12-21) all
# contain a full 7 rush days, so they stay stable under this encoding - it is
# specifically the marked week that moves.
XMAS_PRE_WINDOW = (15, 24)   # December days in the pre-Christmas rush
XMAS_POST_WINDOW = (26, 31)  # December days after Christmas (returns, gift cards)
XMAS_REGRESSOR_COLS = ["xmas_pre_days", "xmas_post_days"]


def _count_december_days(week_end: pd.Timestamp, first_day: int, last_day: int) -> int:
    """Days in the 7-day week ending `week_end` falling in Dec[first_day..last_day]."""
    days = pd.date_range(end=week_end, periods=7, freq="D")
    return int(((days.month == 12) & (days.day >= first_day) & (days.day <= last_day)).sum())


def add_xmas_regressors(dates) -> pd.DataFrame:
    """Date-indexed frame of the two Christmas-overlap regressors.

    Computable for any future date (it only reads the calendar), so the same
    function serves both fitting and prediction.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).unique().sort_values()
    return pd.DataFrame(
        {
            "xmas_pre_days": [_count_december_days(d, *XMAS_PRE_WINDOW) for d in idx],
            "xmas_post_days": [_count_december_days(d, *XMAS_POST_WINDOW) for d in idx],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProphetConfig:
    """Hyperparameters shared by every per-(Store, Dept) fit.

    The defaults are deliberately conservative. Each series is ~143 weekly
    points and the test horizon runs 39 weeks past the end of it, so a freely
    extrapolating piecewise-linear trend is the easiest way to turn a decent
    seasonal fit into a bad forecast. `growth="flat"` drops the trend term
    entirely (level + seasonality + holidays); `"linear"` with a small
    `changepoint_prior_scale` is the damped alternative. Both are worth
    A/B-ing on a holiday-containing fold - see the notebook.

    `seasonality_mode="additive"` is the safe default here too: many
    (Store, Dept) series have a small or occasionally negative level
    (Weekly_Sales includes returns), and multiplicative seasonality on those
    is unstable.

    `use_xmas_regressors` swaps the on/off Christmas holiday dummies for the
    day-count regressors above. Build the holidays frame to match with
    `build_walmart_holidays(include_christmas_dummies=not cfg.use_xmas_regressors)`
    so the two encodings are never fit at the same time (they are collinear).
    """

    growth: str = "flat"
    seasonality_mode: str = "additive"
    yearly_fourier_order: int = 6
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    use_xmas_regressors: bool = True

    def as_mlflow_params(self) -> dict:
        return {f"prophet_{k}": v for k, v in self.__dict__.items()}


def build_walmart_holidays(include_christmas_dummies: bool = False) -> pd.DataFrame:
    """Prophet `holidays=` frame for the 4 Walmart holidays.

    `HOLIDAY_DATES` are already the Friday week-ending dates this dataset
    uses, so they land exactly on the weekly `ds` grid - no window guessing
    needed for Super Bowl, Labor Day or Thanksgiving.

    Christmas is the documented exception (see the module docstring and the
    EDA): the dataset marks the LAST week of the year, but the sales peak is
    the week before. With `include_christmas_dummies=True` this is encoded the
    old way, as two separate dummy events (the marked week and a synthetic
    week-51 peak) so Prophet can learn two coefficients. That encoding is
    still wrong about the year-to-year shift, which is what
    XMAS_REGRESSOR_COLS exists to fix, so the default leaves Christmas out of
    this frame entirely and lets the day-count regressors carry it.
    """
    from src.features import HOLIDAY_DATES

    rows = []
    for name, year_map in HOLIDAY_DATES.items():
        if name == "christmas" and not include_christmas_dummies:
            continue
        for _year, date_str in year_map.items():
            marked_date = pd.Timestamp(date_str)
            rows.append({"holiday": name, "ds": marked_date, "lower_window": 0, "upper_window": 0})
            if name == "christmas":
                rows.append({
                    "holiday": "christmas_peak_week51",
                    "ds": marked_date - pd.Timedelta(weeks=1),
                    "lower_window": 0,
                    "upper_window": 0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _fit_one_series(key, dates, values, holidays_df, config, xmas_frame):
    """Fit one (Store, Dept) Prophet model in a joblib worker process.

    Returns (key, model_json) - a JSON string rather than the live model,
    since that is what survives the trip back to the parent process and a
    later joblib.dump / MLflow artifact reliably.
    """
    from prophet import Prophet
    from prophet.serialize import model_to_json

    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    if config.use_xmas_regressors:
        for col in XMAS_REGRESSOR_COLS:
            df[col] = df["ds"].map(xmas_frame[col])

    model = Prophet(
        growth=config.growth,
        holidays=holidays_df,
        yearly_seasonality=config.yearly_fourier_order,
        # The data is already weekly-aggregated, so there is no intra-week or
        # intra-day pattern for these to capture.
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=config.seasonality_mode,
        changepoint_prior_scale=config.changepoint_prior_scale,
        seasonality_prior_scale=config.seasonality_prior_scale,
        holidays_prior_scale=config.holidays_prior_scale,
        # Prophet otherwise draws 1000 posterior samples on every predict()
        # purely to fill in yhat_lower/yhat_upper. Nothing here reads those,
        # and switching them off is most of what makes ~3,300 series practical.
        uncertainty_samples=0,
    )
    if config.use_xmas_regressors:
        for col in XMAS_REGRESSOR_COLS:
            model.add_regressor(col)
    model.fit(df)
    return key, model_to_json(model)


def fit_series_models(
    train_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    config: ProphetConfig | None = None,
    keys: list[tuple[int, int]] | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[tuple[int, int], str]:
    """Fit one Prophet model per (Store, Dept) series. Returns {key: model_json}.

    `keys` restricts the fit to a subset of (Store, Dept) pairs - used by the
    notebook's config A/B, which compares configurations on a sample of series
    rather than paying for the full panel four times over.

    Series with fewer than MIN_OBS_FOR_PROPHET observations are skipped and
    left to `FallbackStats`.
    """
    from joblib import Parallel, delayed

    config = config or ProphetConfig()
    df = train_df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    xmas_frame = add_xmas_regressors(df["Date"])
    wanted = set(keys) if keys is not None else None

    jobs, skipped = [], 0
    for (store, dept), g in df.groupby(["Store", "Dept"], sort=True):
        key = (int(store), int(dept))
        if wanted is not None and key not in wanted:
            continue
        if len(g) < MIN_OBS_FOR_PROPHET:
            skipped += 1
            continue
        g = g.sort_values("Date")
        jobs.append((key, g["Date"].to_numpy(), g["Weekly_Sales"].to_numpy()))

    if verbose:
        n_cores = os.cpu_count() or 1
        print(
            f"Fitting {len(jobs)} per-(Store, Dept) Prophet models "
            f"(n_jobs={n_jobs}, {n_cores} cores visible); "
            f"{skipped} series under {MIN_OBS_FOR_PROPHET} obs go to the fallback."
        )

    # verbose=10 makes joblib emit a running "N tasks done / elapsed / ETA" line
    # to stderr. Without it a full-panel fit is 20+ silent minutes with no way to
    # tell 10% from 90%, which is not a reasonable thing to ask of anyone.
    results = Parallel(n_jobs=n_jobs, batch_size=16, verbose=10 if verbose else 0)(
        delayed(_fit_one_series)(key, dates, values, holidays_df, config, xmas_frame)
        for key, dates, values in jobs
    )
    if verbose:
        print(f"  {len(results)}/{len(jobs)} series fit.")
    return dict(results)


# ---------------------------------------------------------------------------
# Fallback for series Prophet cannot or should not be fit on
# ---------------------------------------------------------------------------
# A department's seasonal profile is clipped to this range before use. A few
# (Dept, week_of_year) cells are estimated from very few observations and can
# come back absurd; a department genuinely quintupling in one week is already
# at the edge of plausible.
DEPT_INDEX_CLIP = (0.1, 6.0)


@dataclass
class FallbackStats:
    """Estimator for rows no per-series Prophet model covers.

    Two cases: a (Store, Dept) with too little history to fit, and a
    (Store, Dept) that appears in test but never in train at all.

    Both are predicted as `level * dept_woy_index[dept][week_of_year]`: a
    level for that series (its own historical mean, or the department's
    company-wide per-row mean when the pair is unseen) times a normalized
    department-level seasonal profile pooled across all stores and years.
    """

    store_dept_mean: dict[tuple[int, int], float]
    dept_mean: dict[int, float]
    dept_woy_index: dict[int, dict[int, float]]
    global_mean: float


def compute_fallback_stats(train_df: pd.DataFrame) -> FallbackStats:
    """Levels plus a per-department week-of-year seasonal profile.

    Each row is divided by its own (Store, Dept) mean BEFORE pooling, so the
    profile is a shape and not a size - otherwise the largest stores would
    set every department's seasonal curve. The per-cell statistic is a median
    rather than a mean because some (Dept, week_of_year) cells hold only a
    handful of observations across the 2.75 years of history.
    """
    df = train_df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

    store_dept_mean = df.groupby(["Store", "Dept"])["Weekly_Sales"].mean()
    dept_mean = df.groupby("Dept")["Weekly_Sales"].mean()
    global_mean = float(df["Weekly_Sales"].mean())

    level = pd.Series(
        df.set_index(["Store", "Dept"]).index.map(store_dept_mean), index=df.index
    )
    usable = level > 0  # a series averaging <= 0 (net returns) has no meaningful shape
    prof = df.loc[usable].copy()
    prof["norm"] = prof["Weekly_Sales"] / level[usable]

    index = prof.groupby(["Dept", "week_of_year"])["norm"].median().clip(*DEPT_INDEX_CLIP)
    dept_woy_index: dict[int, dict[int, float]] = {}
    for (dept, woy), value in index.items():
        dept_woy_index.setdefault(int(dept), {})[int(woy)] = float(value)

    return FallbackStats(
        store_dept_mean={(int(s), int(d)): float(v) for (s, d), v in store_dept_mean.items()},
        dept_mean={int(d): float(v) for d, v in dept_mean.items()},
        dept_woy_index=dept_woy_index,
        global_mean=global_mean,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _predict_one_series(key, model_json, dates, config, xmas_frame):
    """Forecast one (Store, Dept) series' dates in a joblib worker process."""
    from prophet.serialize import model_from_json

    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    if config.use_xmas_regressors:
        for col in XMAS_REGRESSOR_COLS:
            future[col] = future["ds"].map(xmas_frame[col])

    model = model_from_json(model_json)
    fcast = model.predict(future)
    return key, fcast["ds"].to_numpy(), fcast["yhat"].to_numpy()


class ProphetPanelPipeline:
    """
    Per-(Store, Dept) Prophet panel, exposing the plain .predict(raw_test_df)
    contract CLAUDE.md requires (a custom class with .predict() is fine for
    non-sklearn-compatible per-series statistical models - the same allowance
    src/arima_forecaster.py documents).

    predict() needs only Store, Dept and Date columns from a raw test.csv
    -shaped frame. It does not need IsHoliday: holiday effects are inside the
    per-series Prophet models, learned from the holidays frame and the
    Christmas day-count regressors, both of which are keyed off the date.
    """

    def __init__(
        self,
        series_models_json: dict[tuple[int, int], str],
        fallback: FallbackStats,
        config: ProphetConfig,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        self.series_models_json = series_models_json
        self.fallback = fallback
        self.config = config
        self.n_jobs = n_jobs
        # joblib verbosity for predict(). Deserializing ~3,000 models from JSON
        # and forecasting each is minutes of work, not seconds - worth a counter.
        self.verbose = verbose

    def _fallback_predict(self, df: pd.DataFrame) -> np.ndarray:
        f = self.fallback
        woy = df["Date"].dt.isocalendar().week.astype(int).to_numpy()
        out = np.empty(len(df), dtype=float)
        for i, (store, dept, week) in enumerate(zip(df["Store"], df["Dept"], woy)):
            level = f.store_dept_mean.get((int(store), int(dept)))
            if level is None:
                level = f.dept_mean.get(int(dept), f.global_mean)
            seasonal = f.dept_woy_index.get(int(dept), {}).get(int(week), 1.0)
            out[i] = level * seasonal
        return out

    def predict(self, raw_test_df: pd.DataFrame) -> np.ndarray:
        from joblib import Parallel, delayed

        df = raw_test_df[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Store"] = df["Store"].astype(int)
        df["Dept"] = df["Dept"].astype(int)

        xmas_frame = add_xmas_regressors(df["Date"])

        jobs = []
        for (store, dept), g in df.groupby(["Store", "Dept"], sort=False):
            key = (int(store), int(dept))
            model_json = self.series_models_json.get(key)
            if model_json is None:
                continue
            jobs.append((key, model_json, g["Date"].drop_duplicates().sort_values().to_numpy()))

        results = Parallel(n_jobs=self.n_jobs, batch_size=16, verbose=self.verbose)(
            delayed(_predict_one_series)(key, model_json, dates, self.config, xmas_frame)
            for key, model_json, dates in jobs
        )

        lookup: dict[tuple[int, int, pd.Timestamp], float] = {}
        for (store, dept), dates, yhat in results:
            for date, value in zip(dates, yhat):
                lookup[(store, dept, pd.Timestamp(date))] = float(value)

        preds = np.array(
            [lookup.get(k, np.nan) for k in zip(df["Store"], df["Dept"], df["Date"])],
            dtype=float,
        )

        missing = np.isnan(preds)
        if missing.any():
            preds[missing] = self._fallback_predict(df.loc[missing])

        # Weekly_Sales can be negative in train (returns), but a negative
        # forecast is never the right call and only costs WMAE.
        return np.clip(preds, 0.0, None)

    def coverage(self, raw_test_df: pd.DataFrame) -> dict:
        """What fraction of test rows a fitted Prophet model covers vs the fallback."""
        df = raw_test_df[["Store", "Dept"]].astype(int)
        covered = [
            (s, d) in self.series_models_json for s, d in zip(df["Store"], df["Dept"])
        ]
        n = len(df)
        return {
            "n_rows": n,
            "n_prophet_rows": int(np.sum(covered)),
            "n_fallback_rows": int(n - np.sum(covered)),
            "prophet_row_fraction": float(np.mean(covered)),
            "n_series_fitted": len(self.series_models_json),
        }
