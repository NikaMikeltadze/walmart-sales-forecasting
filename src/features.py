"""
src/features.py

Feature engineering for the Walmart Store Sales Forecasting project.
Builds on the merged/cleaned dataframe produced by src/preprocessing.py.

Expected input columns after preprocessing:
    Store, Dept, Date, IsHoliday, Weekly_Sales (train only),
    Temperature, Fuel_Price, MarkDown1..MarkDown5, CPI, Unemployment,
    Type, Size

All new feature columns are added in place - no columns are dropped here.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Hardcoded Walmart holiday dates, from the competition data dictionary.
# Only years present in train/test are needed (2010-2013).
HOLIDAY_DATES = {
    "super_bowl": {
        2010: "2010-02-12", 2011: "2011-02-11", 2012: "2012-02-10", 2013: "2013-02-08",
    },
    "labor_day": {
        2010: "2010-09-10", 2011: "2011-09-09", 2012: "2012-09-07", 2013: "2013-09-06",
    },
    "thanksgiving": {
        2010: "2010-11-26", 2011: "2011-11-25", 2012: "2012-11-23", 2013: "2013-11-29",
    },
    "christmas": {
        2010: "2010-12-31", 2011: "2011-12-30", 2012: "2012-12-28", 2013: "2013-12-27",
    },
}

LAG_WEEKS = [1, 4, 13, 26, 52]
ROLLING_WEEKS = [4, 13, 52]


def _days_to_holiday(dates: pd.Series, holiday_name: str) -> pd.Series:
    """Signed distance in days from each date to that year's holiday date."""
    years = dates.dt.year
    holiday_map = HOLIDAY_DATES[holiday_name]
    holiday_dates = years.map(lambda y: holiday_map.get(y))
    holiday_dates = pd.to_datetime(holiday_dates)
    return (holiday_dates - dates).dt.days


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and holiday-distance features. Safe for train and test - no leakage risk."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    df["days_to_super_bowl"] = _days_to_holiday(df["Date"], "super_bowl")
    df["days_to_labor_day"] = _days_to_holiday(df["Date"], "labor_day")
    df["days_to_thanksgiving"] = _days_to_holiday(df["Date"], "thanksgiving")
    df["days_to_christmas"] = _days_to_holiday(df["Date"], "christmas")

    return df


def _lookup_pivot(pivot: pd.DataFrame, stores: pd.Series, depts: pd.Series, dates: pd.Series) -> np.ndarray:
    """
    Look up values in a Date x (Store, Dept) pivot table for arbitrary
    (store, dept, date) triples. Returns NaN where the date or the
    store-dept column doesn't exist in the pivot, or where the nearest
    available date is more than 7 days from the requested date.
    """
    out = np.full(len(stores), np.nan)
    pivot_index = pivot.index
    pivot_columns = pivot.columns

    for i, (store, dept, date) in enumerate(zip(stores.values, depts.values, dates.values)):
        col = (store, dept)
        if col not in pivot_columns:
            continue
        idx = pivot_index.searchsorted(date, side="right") - 1
        if idx < 0:
            continue
        actual_date = pivot_index[idx]
        if abs((pd.Timestamp(date) - actual_date).days) > 7:
            continue
        out[i] = pivot.iloc[idx][col]

    return out


class WalmartFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds temporal, lag, rolling, and store/dept aggregate features.

    Must be fit on the TRAIN set only - it stores the Weekly_Sales history
    and store/dept aggregates from whatever it's fit on. transform() can
    then be called on train (for CV) or test (for submission).

    IMPORTANT - read this before trusting lag/rolling features on test:
    Weekly_Sales lags/rolling stats are computed from a history table built
    at fit time. Test spans about 39 weeks past the end of train. For an
    early test week, a lag-1 or lag-4 feature is still inside the fit
    history and is fine. For a LATE test week, a lag-1 or lag-4 feature
    would require a Weekly_Sales value that doesn't exist yet - it's in
    the future, i.e. another row of test. Those cells come back as NaN.

    Three ways to handle this, pick based on your model:
      1. Tree models (XGBoost/LightGBM): let them split on the NaN
         natively (both libraries handle missing values), or impute with
         store_dept_mean_sales. Don't fill with 0 - that biases splits.
      2. Recursive inference: predict test week by week, feed each
         prediction back in with update_history(), then transform the
         next week. Slower but keeps lags real instead of imputed.
      3. Only trust lag_52 for far-horizon test rows (it always reaches
         back into train) and drop lag_1/4/13/26 there.
    This class doesn't pick one for you - NaNs are left as real NaN so you
    can make that call per model. update_history() is there if you want
    recursive inference.
    """

    def __init__(self, lag_weeks=None, rolling_weeks=None):
        self.lag_weeks = lag_weeks if lag_weeks is not None else LAG_WEEKS
        self.rolling_weeks = rolling_weeks if rolling_weeks is not None else ROLLING_WEEKS

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])

        if "Weekly_Sales" not in X.columns:
            raise ValueError(
                "WalmartFeatureEngineer must be fit on the train set, "
                "which needs a Weekly_Sales column."
            )

        # History used for lag / rolling lookups. Keyed by Store, Dept, Date.
        self.history_ = (
            X[["Store", "Dept", "Date", "Weekly_Sales"]]
            .drop_duplicates(subset=["Store", "Dept", "Date"])
            .sort_values(["Store", "Dept", "Date"])
            .reset_index(drop=True)
        )

        # Store/dept/store-dept aggregates, computed on TRAIN only.
        self.store_mean_ = X.groupby("Store")["Weekly_Sales"].mean()
        self.dept_mean_ = X.groupby("Dept")["Weekly_Sales"].mean()
        self.store_dept_mean_ = X.groupby(["Store", "Dept"])["Weekly_Sales"].mean()
        self.global_mean_ = X["Weekly_Sales"].mean()

        return self

    def update_history(self, new_rows: pd.DataFrame):
        """
        Append newly predicted or newly known Weekly_Sales rows to the
        history table. Use this for recursive/step-ahead test inference:
        predict week t, call update_history() with those predictions as
        Weekly_Sales, then transform week t+1.
        new_rows needs columns: Store, Dept, Date, Weekly_Sales.
        """
        new_rows = new_rows[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
        new_rows["Date"] = pd.to_datetime(new_rows["Date"])
        self.history_ = (
            pd.concat([self.history_, new_rows], ignore_index=True)
            .drop_duplicates(subset=["Store", "Dept", "Date"], keep="last")
            .sort_values(["Store", "Dept", "Date"])
            .reset_index(drop=True)
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = add_temporal_features(X)

        # --- store / dept / store-dept aggregates ---
        X["store_mean_sales"] = X["Store"].map(self.store_mean_)
        X["dept_mean_sales"] = X["Dept"].map(self.dept_mean_)
        X["store_dept_mean_sales"] = X.set_index(["Store", "Dept"]).index.map(
            self.store_dept_mean_
        )
        # fall back to global mean for any store/dept combo never seen in train
        X["store_mean_sales"] = X["store_mean_sales"].fillna(self.global_mean_)
        X["dept_mean_sales"] = X["dept_mean_sales"].fillna(self.global_mean_)
        X["store_dept_mean_sales"] = X["store_dept_mean_sales"].fillna(self.global_mean_)

        # --- lag and rolling features, per Store-Dept, from history_ ---
        X = self._add_lag_and_rolling(X)

        return X

    def _add_lag_and_rolling(self, X: pd.DataFrame) -> pd.DataFrame:
        history = self.history_

        # Pivot history to a Date x (Store, Dept) sales table for fast lookups.
        pivot = history.pivot_table(
            index="Date", columns=["Store", "Dept"], values="Weekly_Sales"
        )
        pivot = pivot.sort_index()

        for lag in self.lag_weeks:
            colname = f"sales_lag_{lag}"
            lookup_dates = X["Date"] - pd.Timedelta(weeks=lag)
            X[colname] = _lookup_pivot(pivot, X["Store"], X["Dept"], lookup_dates)

        for window in self.rolling_weeks:
            roll_mean = pivot.rolling(window=window, min_periods=1).mean()
            roll_std = pivot.rolling(window=window, min_periods=2).std()

            # rolling stats are "as of the most recent known week before
            # this row's date" - i.e. shifted back 1 week to avoid leakage
            X[f"sales_rolling_mean_{window}"] = _lookup_pivot(
                roll_mean, X["Store"], X["Dept"], X["Date"] - pd.Timedelta(weeks=1)
            )
            X[f"sales_rolling_std_{window}"] = _lookup_pivot(
                roll_std, X["Store"], X["Dept"], X["Date"] - pd.Timedelta(weeks=1)
            )

        return X