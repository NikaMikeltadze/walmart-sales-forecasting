"""
src/arima_forecaster.py

A lightweight, per-series forecaster used to turn the ARIMA/SARIMA experiment
into a full-coverage Kaggle submission. It is deliberately NOT sklearn-compatible
(the seasonal ARIMA models used in the notebook's theory section are per-series
statistical fits, not a single global estimator) - instead it exposes the plain
`.fit(raw_train) -> self` / `.predict(raw_test) -> np.ndarray` contract that
`CLAUDE.md` (Pipeline & submission requirements) and `docs/person_a_prompt.md`
allow for the classical-model track: "a plain class with .predict(raw_test_df)
is fine here since it's not sklearn-compatible".

Strategy (documented, fast, and theoretically clean):
  forecast = smoothed seasonal-naive anchor (average of y_{t-53}, y_{t-52},
  y_{t-51}) + non-seasonal ARIMA on the deseasonalized residual, clipped at 0.

Why this shape:
  - The seasonal anchor y_{t-52} captures the strong yearly retail seasonality
    that a plain non-seasonal ARIMA would miss over the ~39-week test horizon.
    Lag-52 always reaches back into the training period for every test week
    (test is <=39 weeks past train's end), so the anchor is always defined.
  - The anchor is a 3-point average around lag-52 (window=1: t-53, t-52,
    t-51) rather than the single point y_{t-52}. This was benchmarked on a
    250-series sample of the shared holdout fold: it reduced holdout WMAE
    from 1403.32 (exact single-point anchor) to 1379.72 (~1.7%), because a
    single week's value one year ago is noisy and the average of the 3
    surrounding weeks is a steadier estimate of "what that time of year looks
    like". Wider windows were tested too (window=2 -> 1438, window=3 -> 1507)
    and made things worse - they smear in weeks whose seasonal position is
    genuinely different, diluting the signal rather than just its noise. A
    log1p transform of the series before fitting was also benchmarked and
    made results slightly worse (1432 vs 1403), so it is deliberately not used.
  - Modeling the residual r_t = y_t - y_{t-52} with a small non-seasonal
    `pmdarima.auto_arima` is the fast alternative to a full seasonal ARIMA
    (m=52) fit per series, which would be prohibitively slow across ~3,000
    Store/Dept series. It decomposes cleanly: seasonal component + a stationary
    residual modeled by ARIMA. Richer ARIMA orders (max_p=3, max_q=3, max_d=2)
    were also benchmarked and did not meaningfully improve on the defaults
    (max_p=2, max_q=2, max_d=1), so the defaults stay small for speed.

Performance:
  The ~2,950 per-series `auto_arima` searches are the expensive step. They are
  independent, so `predict` fans them out across CPU cores with joblib
  (`n_jobs`, default all cores). Anchor/fallback logic stays in the main process
  and only the residual-ARIMA fit+forecast is parallelized. This is a pure
  CPU-bound workload (no GPU benefit), so more cores is the only lever - Colab's
  free tier has fewer cores than a typical laptop and will not be faster.

Fallbacks (guarantee full coverage, never NaN):
  - Series shorter than `min_history` weeks, too few residual observations, or
    an auto_arima failure  -> pure seasonal-naive (the lag-52 anchor alone).
  - A missing anchor date (series gap)                            -> that series' own mean.
  - A (Store, Dept) pair unseen in train (new department in test) -> store mean, then global mean.

`pmdarima` is imported lazily inside the worker so that importing this module
never fails even in an environment where pmdarima's binary wheel is broken -
the class stays importable (and picklable) regardless.

No em dashes anywhere (project code-style rule) - hyphens only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _arima_forecast_job(values, horizon, max_p, max_q, max_d, auto_arima_kwargs):
    """Fit a small non-seasonal auto_arima on a residual series and forecast
    `horizon` steps. Module-level (picklable) so joblib can dispatch it to worker
    processes. Returns a float array, or None on any failure (the caller then
    falls back to the pure seasonal-naive anchor)."""
    try:
        import pmdarima as pm

        model = pm.auto_arima(
            values,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            **auto_arima_kwargs,
        )
        forecast = np.asarray(model.predict(n_periods=horizon), dtype=float)
        if forecast.shape[0] != horizon or not np.all(np.isfinite(forecast)):
            return None
        return forecast
    except Exception:
        return None


class ArimaForecaster:
    """Per-series seasonal-naive + residual-ARIMA forecaster for raw test frames.

    Parameters
    ----------
    m:
        Seasonal period in weeks. 52 for weekly data with yearly seasonality.
    anchor_window:
        Number of weeks on each side of the lag-`m` point to average into the
        seasonal anchor (e.g. window=1 averages y_{t-m-1}, y_{t-m}, y_{t-m+1}).
        Benchmarked as the best setting on a holdout sample - window=0 (exact
        single point) is noisier, window>=2 over-smooths and is worse.
    min_history:
        Minimum number of weekly observations a series needs before a residual
        ARIMA is attempted. Shorter series fall back to the seasonal-naive anchor.
    min_resid_obs:
        Minimum number of (non-NaN) residual observations required to fit the
        residual ARIMA. Below this, fall back to the seasonal-naive anchor.
    max_p, max_q, max_d:
        Order caps handed to `pmdarima.auto_arima`. Kept small so the per-series
        stepwise search stays fast across thousands of series.
    residual_arima:
        If False, skip the ARIMA correction entirely and return the pure
        seasonal-naive (lag-52) forecast. Useful as a fast baseline / ablation.
    n_jobs:
        Number of parallel worker processes for the per-series ARIMA fits in
        `predict` (joblib). -1 = all cores (default). 1 = serial.
    anchor_tolerance_days:
        Tolerance for the lag-52 date lookup, to survive small gaps/offsets in a
        series' weekly index.
    auto_arima_kwargs:
        Extra keyword arguments forwarded to `pmdarima.auto_arima`.
    verbose:
        If True, print the number of ARIMA jobs dispatched and a done message.
    """

    def __init__(
        self,
        m: int = 52,
        anchor_window: int = 1,
        min_history: int = 104,
        min_resid_obs: int = 12,
        max_p: int = 2,
        max_q: int = 2,
        max_d: int = 1,
        residual_arima: bool = True,
        n_jobs: int = -1,
        anchor_tolerance_days: int = 3,
        auto_arima_kwargs: dict | None = None,
        verbose: bool = True,
    ):
        self.m = m
        self.anchor_window = anchor_window
        self.min_history = min_history
        self.min_resid_obs = min_resid_obs
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.residual_arima = residual_arima
        self.n_jobs = n_jobs
        self.anchor_tolerance_days = anchor_tolerance_days
        self.auto_arima_kwargs = auto_arima_kwargs or {}
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, raw_train: pd.DataFrame) -> "ArimaForecaster":
        """Cache per-(Store, Dept) weekly history and fallback aggregates.

        `raw_train` is a raw `train.csv`-shaped frame - only Store, Dept, Date,
        and Weekly_Sales are needed (no external merge required)."""
        required = {"Store", "Dept", "Date", "Weekly_Sales"}
        missing = required - set(raw_train.columns)
        if missing:
            raise ValueError(f"fit() requires columns {sorted(required)}, missing {sorted(missing)}")

        df = raw_train[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        self.global_mean_ = float(df["Weekly_Sales"].mean())
        self.store_mean_ = df.groupby("Store")["Weekly_Sales"].mean().to_dict()

        self.history_: dict[tuple[int, int], pd.Series] = {}
        self.store_dept_mean_: dict[tuple[int, int], float] = {}
        for (store, dept), g in df.groupby(["Store", "Dept"], sort=False):
            ser = g.set_index("Date")["Weekly_Sales"].sort_index()
            # collapse any duplicate dates (keep the last observation for that week)
            ser = ser[~ser.index.duplicated(keep="last")]
            self.history_[(int(store), int(dept))] = ser
            self.store_dept_mean_[(int(store), int(dept))] = float(ser.mean())

        return self

    # -------------------------------------------------------------- predict
    def predict(self, raw_test: pd.DataFrame) -> np.ndarray:
        """Forecast Weekly_Sales for every row of a raw `test.csv`-shaped frame.

        Returns a finite float array aligned to `raw_test`'s row order."""
        if not hasattr(self, "history_"):
            raise RuntimeError("ArimaForecaster must be fit() before predict().")

        required = {"Store", "Dept", "Date"}
        missing = required - set(raw_test.columns)
        if missing:
            raise ValueError(f"predict() requires columns {sorted(required)}, missing {sorted(missing)}")

        df = raw_test[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Pass 1 (main process): per series, build the seasonal-naive anchor and
        # queue the residual arrays that need an ARIMA fit. Cheap and vectorized.
        plans = []          # one dict per (Store, Dept) group
        jobs = []           # (resid_values, horizon) tuples to fit in parallel
        for (store, dept), g in df.groupby(["Store", "Dept"], sort=False):
            test_dates = sorted(pd.unique(g["Date"]))
            anchor, resid_values = self._prepare_series(int(store), int(dept), test_dates)
            plan = {
                "row_index": g.index,
                "row_dates": g["Date"].to_numpy(),
                "dates": test_dates,
                "anchor": anchor,
                "job": None,
            }
            if resid_values is not None:
                plan["job"] = len(jobs)
                jobs.append((resid_values, len(test_dates)))
            plans.append(plan)

        # Pass 2: fan the independent ARIMA fits out across cores.
        if jobs:
            from joblib import Parallel, delayed

            if self.verbose:
                print(f"Fitting residual ARIMA on {len(jobs)} series (n_jobs={self.n_jobs}) ...")
            forecasts = Parallel(n_jobs=self.n_jobs)(
                delayed(_arima_forecast_job)(
                    values, horizon, self.max_p, self.max_q, self.max_d, self.auto_arima_kwargs
                )
                for values, horizon in jobs
            )
            if self.verbose:
                n_ok = sum(f is not None for f in forecasts)
                print(f"  {n_ok}/{len(jobs)} residual ARIMA fits succeeded "
                      f"({len(jobs) - n_ok} fell back to seasonal-naive).")
        else:
            forecasts = []

        # Pass 3: assemble anchor + residual forecast, map back to row order.
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for plan in plans:
            resid_forecast = forecasts[plan["job"]] if plan["job"] is not None else None
            if resid_forecast is None:
                preds_sorted = np.clip(plan["anchor"], 0.0, None)
            else:
                preds_sorted = np.clip(plan["anchor"] + resid_forecast, 0.0, None)
            date_to_pred = dict(zip(plan["dates"], preds_sorted))
            out.loc[plan["row_index"]] = pd.Series(plan["row_dates"]).map(date_to_pred).to_numpy()

        # Safety net: nothing should be NaN, but guarantee it for the submission.
        out = out.fillna(self.global_mean_).clip(lower=0.0)
        return out.to_numpy()

    # ---------------------------------------------------------- internals
    def _prepare_series(self, store: int, dept: int, test_dates: list):
        """Return (anchor_array, resid_values_or_None) for one series.

        `anchor_array` is the seasonal-naive lag-52 forecast aligned to sorted
        `test_dates`. `resid_values` is the deseasonalized residual history to
        fit an ARIMA on, or None when the series is too short / unseen (caller
        then uses the anchor alone)."""
        ser = self.history_.get((store, dept))

        # Level used when even the seasonal anchor is unavailable.
        level = self.store_dept_mean_.get((store, dept))
        if level is None or not np.isfinite(level):
            level = self.store_mean_.get(store, self.global_mean_)

        if ser is None or len(ser) == 0:
            return np.full(len(test_dates), max(0.0, float(level))), None

        series_mean = float(ser.mean())
        lag = pd.Timedelta(weeks=self.m)

        anchor = np.empty(len(test_dates), dtype=float)
        for i, d in enumerate(test_dates):
            # Smoothed anchor: average the available weeks within anchor_window
            # of the lag-m point, instead of trusting a single noisy week.
            values = []
            for offset in range(-self.anchor_window, self.anchor_window + 1):
                v = self._lookup(ser, d - lag + pd.Timedelta(weeks=offset))
                if v is not None:
                    values.append(v)
            anchor[i] = float(np.mean(values)) if values else series_mean

        resid_values = None
        if self.residual_arima and len(ser) >= self.min_history:
            resid = (ser - ser.shift(self.m)).dropna()
            if len(resid) >= self.min_resid_obs:
                resid_values = resid.to_numpy()

        return anchor, resid_values

    def _lookup(self, ser: pd.Series, target: pd.Timestamp):
        """Look up a value at `target` in a sorted-by-date Series, tolerant of
        small gaps. Returns a float or None if nothing is within tolerance."""
        pos = ser.index.get_indexer(
            [target], method="nearest", tolerance=pd.Timedelta(days=self.anchor_tolerance_days)
        )
        if pos[0] == -1:
            return None
        return float(ser.iloc[pos[0]])
