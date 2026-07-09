"""
src/pipeline.py

Reusable plumbing that wires src/preprocessing.py + src/features.py into an
sklearn Pipeline accepting raw, unpreprocessed train.csv/test.csv-shaped
data directly. Shared by the XGBoost and LightGBM experiment notebooks so
this harness is only built once.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from src.evaluation import weighted_mae
from src.features import WalmartFeatureEngineer
from src.validation import Split, describe_split, split_frame

MACRO_COLS = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]


class ModelFeatureSelector(BaseEstimator, TransformerMixin):
    """Final Pipeline step before the model.

    Always drops Date and Weekly_Sales (if present) - the two structural
    non-feature columns. Optionally drops an extra configurable list of
    columns (drop_cols) for ablation experiments (e.g. MarkDown or macro
    columns). Optionally pins a fixed category set on named columns
    (fixed_categories, e.g. {"Type": ["A", "B", "C"]}) so a categorical
    model sees a stable category-to-code mapping at fit and predict time,
    regardless of which values happen to appear in a given slice.
    """

    ALWAYS_DROP = ("Date", "Weekly_Sales")

    def __init__(self, drop_cols=None, fixed_categories=None):
        self.drop_cols = drop_cols
        self.fixed_categories = fixed_categories

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, categories in (self.fixed_categories or {}).items():
            if col in X.columns:
                X[col] = X[col].astype("category").cat.set_categories(categories)
        drop = set(self.ALWAYS_DROP) | set(self.drop_cols or [])
        return X.drop(columns=[c for c in drop if c in X.columns])


def build_cv_frames(
    train_feat: pd.DataFrame,
    splits: list[Split],
    feature_engineer_kwargs: dict | None = None,
) -> list[dict]:
    """Precompute fully-featurized (train, val) frames per CV fold, once.

    train_feat must already have gone through WalmartPreprocessor.transform()
    and add_temporal_features() - both stateless (not target-dependent), so
    safe to run globally once before calling this.

    A FRESH WalmartFeatureEngineer is fit per fold on that fold's train rows
    only (dates <= split.train_end). This is required for correctness: its
    store/dept aggregates and lag/rolling history are plain aggregates over
    whatever is passed to fit(), so fitting once on the full train set and
    reusing it across folds would leak each fold's own validation-period
    sales into the aggregate features used to predict that same fold.

    Returns a list of {"split", "train", "val", "feature_engineer"} dicts,
    one per fold, in fold order.
    """
    kwargs = feature_engineer_kwargs or {}
    frames = []
    for split in splits:
        fold_train, fold_val = split_frame(train_feat, split)
        fe = WalmartFeatureEngineer(**kwargs).fit(fold_train)
        frames.append(
            {
                "split": split,
                "train": fe.transform(fold_train),
                "val": fe.transform(fold_val),
                "feature_engineer": fe,
            }
        )
    return frames


def run_cv_experiment(
    estimator,
    cv_frames: list[dict],
    drop_cols=None,
    fixed_categories=None,
    sample_weight_fn=None,
) -> dict:
    """Fit a clone of `estimator` per cached fold, score with WMAE and MAE.

    sample_weight_fn, if given, is called as sample_weight_fn(fold_train_df)
    -> ndarray and passed to estimator.fit() as sample_weight.

    Returns {"fold_results": [...], "wmae_mean", "wmae_std", "mae_mean"}.
    """
    selector = ModelFeatureSelector(drop_cols=drop_cols, fixed_categories=fixed_categories)
    fold_results = []
    for f in cv_frames:
        X_train = selector.fit_transform(f["train"])
        y_train = f["train"]["Weekly_Sales"]
        X_val = selector.transform(f["val"])
        y_val = f["val"]["Weekly_Sales"]

        model = clone(estimator)
        fit_kwargs = {}
        if sample_weight_fn is not None:
            fit_kwargs["sample_weight"] = sample_weight_fn(f["train"])
        model.fit(X_train, y_train, **fit_kwargs)

        pred = model.predict(X_val)
        wmae = weighted_mae(y_val.values, pred, f["val"]["IsHoliday"].values)
        mae = float(np.mean(np.abs(y_val.values - pred)))
        fold_results.append({**describe_split(f["split"]), "wmae": wmae, "mae": mae})

    wmaes = [r["wmae"] for r in fold_results]
    return {
        "fold_results": fold_results,
        "wmae_mean": float(np.mean(wmaes)),
        "wmae_std": float(np.std(wmaes)),
        "mae_mean": float(np.mean([r["mae"] for r in fold_results])),
    }
