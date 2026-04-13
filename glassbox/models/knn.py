"""
glassbox.models.knn
---------------------
K-Nearest Neighbors for classification and regression.
Supports Euclidean and Manhattan distance metrics.
"""

import numpy as np
from glassbox.utils import pairwise_distances


class KNearestNeighbors:
    """
    K-Nearest Neighbors.

    Parameters
    ----------
    k : int             Number of neighbors.
    task : str          'classification' | 'regression'
    metric : str        'euclidean' | 'manhattan'
    """

    def __init__(self, k: int = 5, task: str = "classification", metric: str = "euclidean"):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.task = task
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self.X_train_ = X.astype(float)
        self.y_train_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = X.astype(float)
        dists = pairwise_distances(X, self.X_train_, metric=self.metric)
        preds = []
        for row in dists:
            nn_idx = np.argsort(row)[: self.k]
            neighbors = self.y_train_[nn_idx]
            if self.task == "classification":
                vals, counts = np.unique(neighbors, return_counts=True)
                preds.append(vals[np.argmax(counts)])
            else:
                preds.append(float(np.mean(neighbors)))
        return np.array(preds)
