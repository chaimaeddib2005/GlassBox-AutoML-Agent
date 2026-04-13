"""
glassbox.optimization.cross_validation
----------------------------------------
K-Fold Cross-Validation splitter — pure NumPy.
"""

import numpy as np


class KFoldCV:
    """
    K-Fold Cross-Validation splitter.

    Parameters
    ----------
    k : int         Number of folds (default 5).
    shuffle : bool  Shuffle data before splitting.
    random_state : int   Seed for shuffle.

    Usage
    -----
    >>> kf = KFoldCV(k=5)
    >>> for X_train, y_train, X_val, y_val in kf.split(X, y):
    ...     model.fit(X_train, y_train)
    ...     score = evaluate(model, X_val, y_val)
    """

    def __init__(self, k: int = 5, shuffle: bool = True, random_state: int = 42):
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Generator yielding (X_train, y_train, X_val, y_val) for each fold.
        """
        n = len(X)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.k, n // self.k, dtype=int)
        fold_sizes[: n % self.k] += 1

        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current: current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            current += fold_size

    def cross_val_score(self, model_class, params: dict, X: np.ndarray, y: np.ndarray, score_fn) -> np.ndarray:
        """
        Run K-fold CV and return array of fold scores.

        Parameters
        ----------
        model_class : class     Uninstantiated model.
        params : dict           Init parameters for model.
        X, y : np.ndarray       Data.
        score_fn : callable     score_fn(y_true, y_pred) -> float
        """
        scores = []
        for X_tr, y_tr, X_val, y_val in self.split(X, y):
            model = model_class(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(score_fn(y_val, preds))
        return np.array(scores)
