"""Shared evaluation metrics for all models"""

import numpy as np


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> float:
    """WMAE: holiday weeks are weighted 5x since they're penalized harder by the competition."""
    weights = np.where(is_holiday, 5.0, 1.0)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
