"""
src/prophet_forecaster.py

Per-store Prophet forecaster used to turn the Prophet experiment into a
full-coverage Kaggle submission without fitting one of Facebook Prophet's
per-series models for each of the ~3,000 (Store, Dept) pairs. Unlike the
darts models (DLinear/N-BEATS/TFT), Prophet has no global/panel training
mode - "training Prophet" always means fitting one model per series, so
fitting all ~3,000 series would blow the time budget for this notebook.

Instead this fits one Prophet model per Store (45 series - the same
"aggregate level" scope the person_b guide recommends, at store granularity
rather than a single company-wide series), then recovers (Store, Dept)-level
Weekly_Sales by scaling each store's forecast by that department's
historical share of the store's total sales.

The per-store fit worker (_fit_one_store_prophet) is a module-level function
so joblib's loky backend can reliably pickle/dispatch it to worker processes
- same reasoning src/arima_forecaster.py's _arima_forecast_job documents for
its own per-series worker.

Prophet models are serialized with prophet.serialize.model_to_json /
model_from_json (a plain JSON string) rather than raw pickle - this round
-trips through joblib.dump/mlflow artifact logging far more reliably than a
cmdstanpy/Stan-backed object would.

No em dashes anywhere (project code-style rule) - hyphens only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features import add_temporal_features

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Store/Date-level columns from features.csv usable as Prophet regressors
# (Temperature/Fuel_Price/CPI/Unemployment/MarkDown1-5). Unlike the per-dept
# training data, features.csv is already at Store+Date grain - exactly what
# the per-store Prophet models need, no disaggregation required.
EXTERNAL_REGRESSOR_COLS = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
]


def prepare_external_regressors(features_df: pd.DataFrame) -> pd.DataFrame:
    """Clean features.csv's Store/Date columns for use as Prophet regressors.

    MarkDown NaNs mean "no promotion that week" (same convention as
    `src.preprocessing.clean`) - filled with 0.0, since Prophet regressors
    cannot contain NaN. CPI/Unemployment gaps are already forward-filled by
    `src.preprocessing.load_raw_data`.
    """
    out = features_df[["Store", "Date"] + EXTERNAL_REGRESSOR_COLS].copy()
    markdown_cols = [c for c in EXTERNAL_REGRESSOR_COLS if c.startswith("MarkDown")]
    out[markdown_cols] = out[markdown_cols].fillna(0.0)
    return out


def _fit_one_store_prophet(
    store, dates, values, holidays_df, regressor_frame, regressor_cols, external_frame=None
):
    """Fit one store's Prophet model in a joblib worker process.

    Returns (store, model_json) - a JSON string, not the live model object,
    since that is what survives the round trip back to the main process
    (and later a joblib.dump/mlflow artifact) reliably."""
    from prophet import Prophet
    from prophet.serialize import model_to_json

    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    regressor_cols = regressor_cols or []
    for col in regressor_cols:
        df[col] = df["ds"].map(regressor_frame[col])
    if external_frame is not None:
        for col in EXTERNAL_REGRESSOR_COLS:
            df[col] = df["ds"].map(external_frame[col])

    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        # Holiday/seasonal swings scale with a store's baseline volume rather
        # than adding a fixed dollar amount - multiplicative fits that better
        # than Prophet's additive default for this kind of retail series.
        seasonality_mode="multiplicative",
    )
    for col in regressor_cols:
        model.add_regressor(col)
    if external_frame is not None:
        for col in EXTERNAL_REGRESSOR_COLS:
            model.add_regressor(col)
    model.fit(df)
    return int(store), model_to_json(model)


def compute_store_share_stats(train_df: pd.DataFrame) -> tuple[dict[int, float], float]:
    """Mean per-date STORE TOTAL Weekly_Sales (summed across depts), per store,
    plus the grand mean of those store totals across all stores/dates.

    This is the correct denominator for `ProphetForecastPipeline`'s dept-share
    calculation, since the per-store Prophet model is fit on that same
    per-date store total (see `fit_store_models` below).
    `WalmartFeatureEngineer.store_mean_` is a *per-row* mean instead (averaged
    over every (Store, Dept, Date) row), which is smaller than the store's
    total by roughly a factor of the store's department count and cannot be
    substituted here - doing so inflates every disaggregated (Store, Dept)
    prediction by that same factor.
    """
    store_totals = train_df.groupby(["Store", "Date"])["Weekly_Sales"].sum()
    store_total_mean = store_totals.groupby("Store").mean().to_dict()
    avg_store_total = float(store_totals.mean())
    return store_total_mean, avg_store_total


def fit_store_models(
    train_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    regressor_cols: list[str] | None = None,
    external_features: pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[int, str]:
    """Fit one Prophet model per Store on a raw (Store, Dept, Date,
    Weekly_Sales) frame, aggregated to store level first. The ~45
    independent fits are fanned out across CPU cores via joblib - this is
    what keeps store-level Prophet fast enough to fit the notebook's time
    budget (a handful of extra linear regressors per fit costs
    milliseconds, not minutes). Returns dict[store] -> model_json.

    `external_features` is an optional `prepare_external_regressors(...)`
    -shaped frame (Store, Date, EXTERNAL_REGRESSOR_COLS) - pass the same
    frame to `ProphetForecastPipeline` so predict() adds the same regressors.
    """
    from joblib import Parallel, delayed

    store_totals = train_df.groupby(["Store", "Date"])["Weekly_Sales"].sum().reset_index()
    regressor_frame = add_temporal_features(
        store_totals[["Date"]].drop_duplicates()
    ).set_index("Date")

    external_by_store = None
    if external_features is not None:
        external_by_store = {
            store: g.set_index("Date") for store, g in external_features.groupby("Store")
        }

    jobs = [
        (store, g["Date"].to_numpy(), g["Weekly_Sales"].to_numpy())
        for store, g in store_totals.groupby("Store")
    ]
    if verbose:
        print(f"Fitting {len(jobs)} store-level Prophet models (n_jobs={n_jobs}) ...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_store_prophet)(
            store, dates, values, holidays_df, regressor_frame, regressor_cols,
            external_by_store[store] if external_by_store is not None else None,
        )
        for store, dates, values in jobs
    )
    if verbose:
        print(f"  {len(results)}/{len(jobs)} store models fit.")
    return dict(results)


class ProphetForecastPipeline:
    """
    Store-level Prophet forecaster + dept-level disaggregation, exposing the
    plain .predict(raw_test_df) contract required by CLAUDE.md (a custom
    class with .predict() is fine for non-sklearn-compatible per-series
    statistical models - same allowance src/arima_forecaster.py documents).

    `external_features` (optional) is a `prepare_external_regressors(...)`
    -shaped frame (Store, Date, Temperature/Fuel_Price/CPI/Unemployment/
    MarkDown1-5) covering both the train and test date ranges - features.csv
    already spans both, so predict() still only needs Store, Dept, Date
    columns from raw test.csv, with the external values looked up internally.
    """

    def __init__(
        self,
        store_models_json: dict[int, str],
        store_total_mean: dict[int, float],
        store_dept_mean: dict[tuple[int, int], float],
        dept_mean: dict[int, float],
        avg_store_total: float,
        global_row_mean: float,
        regressor_cols: list[str] | None = None,
        external_features: pd.DataFrame | None = None,
    ):
        self.store_models_json = store_models_json
        self.store_total_mean = store_total_mean
        self.store_dept_mean = store_dept_mean
        self.dept_mean = dept_mean
        self.avg_store_total = avg_store_total
        self.global_row_mean = global_row_mean
        self.regressor_cols = regressor_cols or []
        self.external_features = external_features

    def _dept_share(self, store: int, dept: int) -> float:
        """(Store, Dept)'s historical share of that store's total sales.

        `store_dept_mean` (a per-dept row mean) divided by `store_total_mean`
        (a per-store TOTAL, summed across depts) gives the actual fraction of
        the store's forecast this dept should receive - see
        `compute_store_share_stats`. Falls back to the department's average
        share of an average store's total (dept_mean / avg_store_total) when
        this exact (Store, Dept) combo was never seen at fit time (e.g. a
        department appearing at a store for the first time in test)."""
        store_total = self.store_total_mean.get(store, self.avg_store_total)
        sd_mean = self.store_dept_mean.get((store, dept))
        if sd_mean is not None and store_total > 0:
            return sd_mean / store_total
        if self.avg_store_total > 0:
            return self.dept_mean.get(dept, 0.0) / self.avg_store_total
        return 0.0

    def predict(self, raw_test_df: pd.DataFrame) -> np.ndarray:
        """Forecast Weekly_Sales for every row of a raw test.csv-shaped frame."""
        from prophet.serialize import model_from_json

        df = raw_test_df[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Store"] = df["Store"].astype(int)
        df["Dept"] = df["Dept"].astype(int)

        feat = add_temporal_features(df[["Date"]].drop_duplicates()).set_index("Date")

        external_by_store = None
        if self.external_features is not None:
            external_by_store = {
                store: g.set_index("Date")
                for store, g in self.external_features.groupby("Store")
            }

        preds = pd.Series(np.nan, index=df.index, dtype=float)
        for store, model_json in self.store_models_json.items():
            store_mask = df["Store"] == store
            if not store_mask.any():
                continue

            store_dates = df.loc[store_mask, "Date"].drop_duplicates().sort_values()
            future = pd.DataFrame({"ds": store_dates})
            for col in self.regressor_cols:
                future[col] = future["ds"].map(feat[col])
            if external_by_store is not None and store in external_by_store:
                ext = external_by_store[store]
                for col in EXTERNAL_REGRESSOR_COLS:
                    future[col] = future["ds"].map(ext[col])

            model = model_from_json(model_json)
            fcast = model.predict(future)
            yhat_by_date = dict(zip(fcast["ds"], fcast["yhat"]))

            for dept in df.loc[store_mask, "Dept"].unique():
                mask = store_mask & (df["Dept"] == dept)
                share = self._dept_share(store, int(dept))
                preds.loc[mask] = df.loc[mask, "Date"].map(yhat_by_date).to_numpy() * share

        preds = preds.fillna(self.global_row_mean).clip(lower=0.0)
        return preds.to_numpy()
