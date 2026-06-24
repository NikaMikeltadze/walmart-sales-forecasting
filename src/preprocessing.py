"""Shared preprocessing pipeline for all models"""

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
MACRO_COLS = ["CPI", "Unemployment"]

_NESTED_FILES = {"train.csv", "test.csv", "features.csv"}


def _fill_macro_gaps(features: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill CPI/Unemployment per store to cover NaN gaps in the test period.
Reasonable since these are slow-moving macro series"""
    features = features.sort_values(["Store", "Date"]).copy()
    features[MACRO_COLS] = features.groupby("Store")[MACRO_COLS].ffill()
    return features


def load_raw_data(data_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    """Load train, test, features and stores tables from the raw data directory."""
    data_dir = Path(data_dir)
    tables = {}
    for name in ("train.csv", "test.csv", "features.csv"):
        path = data_dir / name / name if name in _NESTED_FILES else data_dir / name
        tables[name.removesuffix(".csv")] = pd.read_csv(path, parse_dates=["Date"])
    tables["features"] = _fill_macro_gaps(tables["features"])
    tables["stores"] = pd.read_csv(data_dir / "stores.csv")
    return tables


def merge_external(df: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Merge a train/test-shaped frame with the features and stores lookup tables."""
    merged = df.merge(features.drop(columns="IsHoliday"), on=["Store", "Date"], how="left")
    merged = merged.merge(stores, on="Store", how="left")
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill MarkDown NaNs (no promotion that week) and normalize dtypes."""
    df = df.copy()
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0.0)
    df["IsHoliday"] = df["IsHoliday"].astype(bool)
    df["Store"] = df["Store"].astype(int)
    df["Dept"] = df["Dept"].astype(int)
    df["Type"] = df["Type"].astype("category")
    return df


class WalmartPreprocessor(BaseEstimator, TransformerMixin):
    """Pipeline step that merges and cleans raw train/test data with features and stores tables.

    Accepts unpreprocessed test.csv-style input; loads features/stores at fit time.
    """

    def __init__(self, data_dir: str | Path = "data/raw"):
        self.data_dir = data_dir

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "WalmartPreprocessor":
        data = load_raw_data(self.data_dir)
        self.features_ = data["features"]
        self.stores_ = data["stores"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        merged = merge_external(X, self.features_, self.stores_)
        return clean(merged)
