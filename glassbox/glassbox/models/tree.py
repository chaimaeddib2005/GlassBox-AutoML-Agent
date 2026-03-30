"""
glassbox.models.tree
---------------------
Decision Tree for classification (Gini Impurity) and regression (MSE Variance Reduction).
Pure recursive NumPy implementation.
"""

import numpy as np


def _gini(y: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - float(np.sum(probs ** 2))


def _mse(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - np.mean(y)) ** 2))


def _best_split(X: np.ndarray, y: np.ndarray, task: str, min_samples_split: int):
    """Find the best (feature_idx, threshold) split."""
    n_features = X.shape[1]
    impurity_fn = _gini if task == "classification" else _mse
    base_impurity = impurity_fn(y)
    best_gain = -np.inf
    best_feat = None
    best_thresh = None

    for feat_idx in range(n_features):
        col = X[:, feat_idx]
        thresholds = np.unique(col)

        for thresh in thresholds:
            left_mask = col <= thresh
            right_mask = ~left_mask
            if left_mask.sum() < min_samples_split or right_mask.sum() < min_samples_split:
                continue

            y_left, y_right = y[left_mask], y[right_mask]
            n = len(y)
            weighted = (len(y_left) / n) * impurity_fn(y_left) + (len(y_right) / n) * impurity_fn(y_right)
            gain = base_impurity - weighted

            if gain > best_gain:
                best_gain = gain
                best_feat = feat_idx
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


class _Node:
    __slots__ = ["feat", "thresh", "left", "right", "value", "n_samples", "impurity"]

    def __init__(self):
        self.feat = None
        self.thresh = None
        self.left = None
        self.right = None
        self.value = None
        self.n_samples = 0
        self.impurity = 0.0


class DecisionTree:
    """
    Decision Tree — supports classification and regression.

    Parameters
    ----------
    task : str            'classification' | 'regression'
    max_depth : int       Maximum depth (None = unlimited).
    min_samples_split : int  Minimum samples to attempt a split.
    criterion : str       'gini' (classification) | 'mse' (regression). Auto-set from task.
    """

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = None,
        min_samples_split: int = 2,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.n_features_ = X.shape[1]
        self._importance_sum = np.zeros(self.n_features_)
        self.root_ = self._build(X, y, depth=0)
        total = self._importance_sum.sum()
        self.feature_importances_ = self._importance_sum / (total if total > 0 else 1)
        return self

    def _build(self, X, y, depth) -> _Node:
        node = _Node()
        node.n_samples = len(y)

        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            node.value = vals[np.argmax(counts)]
            node.impurity = _gini(y)
        else:
            node.value = float(np.mean(y))
            node.impurity = _mse(y)

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split * 2:
            return node

        feat, thresh, gain = _best_split(X, y, self.task, self.min_samples_split)
        if feat is None or gain <= 0:
            return node

        # Track importance (weighted impurity reduction)
        self._importance_sum[feat] += gain * len(y)

        mask = X[:, feat] <= thresh
        node.feat = feat
        node.thresh = thresh
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, x, node: _Node):
        if node.feat is None:
            return node.value
        if x[node.feat] <= node.thresh:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.array([self._predict_one(x, self.root_) for x in X])

    def feature_importance(self, feature_names=None) -> dict:
        if self.feature_importances_ is None:
            return {}
        names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        return dict(sorted(zip(names, self.feature_importances_.tolist()), key=lambda x: -x[1]))
