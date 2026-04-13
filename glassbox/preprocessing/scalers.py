"""
glassbox.preprocessing.scalers
-------------------------------
MinMaxScaler and StandardScaler — pure NumPy implementations.
"""

import numpy as np
from glassbox.utils import mean, std


class MinMaxScaler:
    """
    Scale features to [0, 1].
    x_scaled = (x - x_min) / (x_max - x_min)
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = X.astype(float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X.astype(float) - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self.range_ + self.min_


class StandardScaler:
    """
    Standardize features to zero mean and unit variance.
    x_std = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = X.astype(float)
        self.mean_ = np.array([mean(X[:, i]) for i in range(X.shape[1])])
        self.std_ = np.array([std(X[:, i]) for i in range(X.shape[1])])
        self.std_[self.std_ == 0] = 1  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X.astype(float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_std: np.ndarray) -> np.ndarray:
        return X_std * self.std_ + self.mean_
