"""
Baseline pitch velocity prediction model.
Modify this file to minimize rmse_mph.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def build_model():
    """Build and return an sklearn regressor."""
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )


def extract_features(X):
    """Feature engineering. X is (n_samples, n_raw_features) numpy array."""
    return X
