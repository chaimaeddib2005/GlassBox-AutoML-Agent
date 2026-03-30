"""
glassbox.models.forest
-----------------------
Random Forest — bagging ensemble of DecisionTrees with √features subsampling.
"""

import numpy as np
from glassbox.models.tree import DecisionTree


class RandomForest:
    """
    Random Forest for classification and regression.

    Parameters
    ----------
    n_estimators : int      Number of trees.
    task : str              'classification' | 'regression'
    max_depth : int         Max depth per tree (None = unlimited).
    min_samples_split : int Min samples to split a node.
    max_features : str|int  'sqrt' (default) | 'log2' | int
    random_state : int      Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        task: str = "classification",
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.feature_subsets_ = []
        self.feature_importances_ = None
        self.n_features_ = None

    def _get_n_features(self, p: int) -> int:
        if self.max_features == "sqrt":
            return max(2, int(np.sqrt(p)))
        elif self.max_features == "log2":
            return max(2, int(np.log2(p + 1)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, p)
        return p

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        self.n_features_ = p
        self.trees_ = []
        self.feature_subsets_ = []
        n_feat = self._get_n_features(p)
        importance_acc = np.zeros(p)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = rng.randint(0, n, size=n)
            X_boot, y_boot = X[idx], y[idx]

            # Feature subspace
            feat_idx = rng.choice(p, size=n_feat, replace=False)
            feat_idx = np.sort(feat_idx)

            tree = DecisionTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees_.append(tree)
            self.feature_subsets_.append(feat_idx)

            # Accumulate importances back to original indices
            for local_i, global_i in enumerate(feat_idx):
                importance_acc[global_i] += tree.feature_importances_[local_i]

        total = importance_acc.sum()
        self.feature_importances_ = importance_acc / (total if total > 0 else 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Call fit() before predict().")
        # Each tree predicts on its own feature subset
        preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self.feature_subsets_)
        ])  # shape: (n_estimators, n_samples)

        if self.task == "classification":
            # Majority vote
            result = []
            for i in range(X.shape[0]):
                votes = preds[:, i]
                vals, counts = np.unique(votes, return_counts=True)
                result.append(vals[np.argmax(counts)])
            return np.array(result)
        else:
            return preds.mean(axis=0)

    def feature_importance(self, feature_names=None) -> dict:
        if self.feature_importances_ is None:
            return {}
        names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        return dict(sorted(zip(names, self.feature_importances_.tolist()), key=lambda x: -x[1]))
