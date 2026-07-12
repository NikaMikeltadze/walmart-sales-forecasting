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


def _fit_one_store_prophet(store, dates, values, holidays_df, regressor_frame, regressor_cols):
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

    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    for col in regressor_cols:
        model.add_regressor(col)
    model.fit(df)
    return int(store), model_to_json(model)


def fit_store_models(
    train_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    regressor_cols: list[str] | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[int, str]:
    """Fit one Prophet model per Store on a raw (Store, Dept, Date,
    Weekly_Sales) frame, aggregated to store level first. The ~45
    independent fits are fanned out across CPU cores via joblib - this is
    what keeps store-level Prophet fast enough to fit the notebook's time
    budget. Returns dict[store] -> model_json."""
    from joblib import Parallel, delayed

    store_totals = train_df.groupby(["Store", "Date"])["Weekly_Sales"].sum().reset_index()
    regressor_frame = add_temporal_features(
        store_totals[["Date"]].drop_duplicates()
    ).set_index("Date")

    jobs = [
        (store, g["Date"].to_numpy(), g["Weekly_Sales"].to_numpy())
        for store, g in store_totals.groupby("Store")
    ]
    if verbose:
        print(f"Fitting {len(jobs)} store-level Prophet models (n_jobs={n_jobs}) ...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_store_prophet)(
            store, dates, values, holidays_df, regressor_frame, regressor_cols
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

    No external features (Temperature/CPI/MarkDown/...) are merged in -
    Prophet's regressors here are Date-derived only (add_temporal_features),
    so predict() only needs Store, Dept, Date columns from raw test.csv.
    """

    def __init__(
        self,
        store_models_json: dict[int, str],
        store_mean: dict[int, float],
        store_dept_mean: dict[tuple[int, int], float],
        dept_mean: dict[int, float],
        global_mean: float,
        regressor_cols: list[str] | None = None,
    ):
        self.store_models_json = store_models_json
        self.store_mean = store_mean
        self.store_dept_mean = store_dept_mean
        self.dept_mean = dept_mean
        self.global_mean = global_mean
        self.regressor_cols = regressor_cols or []

    def _dept_share(self, store: int, dept: int) -> float:
        """(Store, Dept)'s historical share of that store's total sales.

        Falls back to the department's average share of company-wide sales
        (dept_mean / global_mean) when this exact (Store, Dept) combo was
        never seen at fit time (e.g. a department appearing at a store for
        the first time in test)."""
        store_mean = self.store_mean.get(store, self.global_mean)
        sd_mean = self.store_dept_mean.get((store, dept))
        if sd_mean is not None and store_mean > 0:
            return sd_mean / store_mean
        return self.dept_mean.get(dept, self.global_mean) / self.global_mean

    def predict(self, raw_test_df: pd.DataFrame) -> np.ndarray:
        """Forecast Weekly_Sales for every row of a raw test.csv-shaped frame."""
        from prophet.serialize import model_from_json

        df = raw_test_df[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Store"] = df["Store"].astype(int)
        df["Dept"] = df["Dept"].astype(int)

        feat = add_temporal_features(df[["Date"]].drop_duplicates()).set_index("Date")

        preds = pd.Series(np.nan, index=df.index, dtype=float)
        for store, model_json in self.store_models_json.items():
            store_mask = df["Store"] == store
            if not store_mask.any():
                continue

            store_dates = df.loc[store_mask, "Date"].drop_duplicates().sort_values()
            future = pd.DataFrame({"ds": store_dates})
            for col in self.regressor_cols:
                future[col] = future["ds"].map(feat[col])

            model = model_from_json(model_json)
            fcast = model.predict(future)
            yhat_by_date = dict(zip(fcast["ds"], fcast["yhat"]))

            for dept in df.loc[store_mask, "Dept"].unique():
                mask = store_mask & (df["Dept"] == dept)
                share = self._dept_share(store, int(dept))
                preds.loc[mask] = df.loc[mask, "Date"].map(yhat_by_date).to_numpy() * share

        preds = preds.fillna(self.global_mean).clip(lower=0.0)
        return preds.to_numpy()
