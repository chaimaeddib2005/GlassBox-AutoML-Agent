"""
glassbox.preprocessing.imputer
-------------------------------
Fills missing values using statistical strategies.
"""

import numpy as np
from glassbox.utils import mean, median, mode


class SimpleImputer:
    """
    Fill missing values (np.nan) in numerical or categorical columns.

    Parameters
    ----------
    strategy : str
        'mean' | 'median' | 'mode'
        For categorical columns, 'mode' is always used regardless of strategy.
    """

    STRATEGIES = ("mean", "median", "mode")

    def __init__(self, strategy: str = "mean"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        self.fill_values_ = []
        for i in range(X.shape[1]):
            col = X[:, i]
            try:
                col_f = col.astype(float)
                valid = col_f[~np.isnan(col_f)]
                if self.strategy == "mean":
                    val = mean(valid)
                elif self.strategy == "median":
                    val = median(valid)
                else:
                    val = mode(valid)
                self.fill_values_.append(float(val))
            except (ValueError, TypeError):
                # categorical
                valid = col[(col != None) & (col != "")]  # noqa: E711
                self.fill_values_.append(mode(valid))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.fill_values_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_out = X.copy().astype(object)
        for i, fill in enumerate(self.fill_values_):
            col = X_out[:, i]
            try:
                mask = np.isnan(col.astype(float))
            except (ValueError, TypeError):
                mask = np.array([v is None or v == "" for v in col])
            col[mask] = fill
            X_out[:, i] = col
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
