"""
glassbox.optimization.search
-----------------------------
Grid Search and Random Search with K-Fold cross-validation scoring.
"""

import numpy as np
import itertools
import time
from glassbox.optimization.cross_validation import KFoldCV
from glassbox.evaluation.metrics import ClassificationMetrics, RegressionMetrics


def _score(model, X, y, task, metric):
    preds = model.predict(X)
    if task == "classification":
        m = ClassificationMetrics(y, preds)
        return getattr(m, metric)()
    else:
        m = RegressionMetrics(y, preds)
        return getattr(m, metric)()


class GridSearch:
    """
    Exhaustive search over a hyperparameter grid.

    Parameters
    ----------
    model_class : class     Uninstantiated model class.
    param_grid : dict       {'param_name': [val1, val2, ...], ...}
    task : str              'classification' | 'regression'
    cv : int                Number of K-Fold splits.
    scoring : str           Metric name on ClassificationMetrics or RegressionMetrics.
    higher_is_better : bool Direction of optimization.

    Example
    -------
    >>> gs = GridSearch(DecisionTree, {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]},
    ...                 task='classification', cv=5, scoring='f1')
    >>> gs.fit(X_train, y_train)
    >>> print(gs.best_params_)
    """

    def __init__(
        self,
        model_class,
        param_grid: dict,
        task: str = "classification",
        cv: int = 5,
        scoring: str = "accuracy",
        higher_is_better: bool = True,
    ):
        self.model_class = model_class
        self.param_grid = param_grid
        self.task = task
        self.cv = cv
        self.scoring = scoring
        self.higher_is_better = higher_is_better
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearch":
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        kfold = KFoldCV(k=self.cv)
        self.results_ = []

        best_score = -np.inf if self.higher_is_better else np.inf
        best_params = None

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            scores = []

            for X_tr, y_tr, X_val, y_val in kfold.split(X, y):
                model = self.model_class(**params)
                model.fit(X_tr, y_tr)
                s = _score(model, X_val, y_val, self.task, self.scoring)
                scores.append(s)

            avg_score = float(np.mean(scores))
            self.results_.append({"params": params, "score": avg_score})

            is_better = avg_score > best_score if self.higher_is_better else avg_score < best_score
            if is_better:
                best_score = avg_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        return self

    def best_model(self, X: np.ndarray, y: np.ndarray):
        """Return a model trained on full data with best_params_."""
        model = self.model_class(**self.best_params_)
        model.fit(X, y)
        return model


class RandomSearch:
    """
    Stochastic hyperparameter search within a time budget or n_iter limit.

    Parameters
    ----------
    model_class : class     Uninstantiated model class.
    param_distributions : dict
        {'param_name': [val1, val2, ...] | np.ndarray of values}
    task, cv, scoring, higher_is_better : same as GridSearch.
    n_iter : int            Max combinations to try.
    time_budget : float     Max seconds to spend (None = no limit).
    random_state : int      Seed for reproducibility.
    """

    def __init__(
        self,
        model_class,
        param_distributions: dict,
        task: str = "classification",
        cv: int = 5,
        scoring: str = "accuracy",
        higher_is_better: bool = True,
        n_iter: int = 20,
        time_budget: float = None,
        random_state: int = 42,
    ):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.task = task
        self.cv = cv
        self.scoring = scoring
        self.higher_is_better = higher_is_better
        self.n_iter = n_iter
        self.time_budget = time_budget
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearch":
        rng = np.random.RandomState(self.random_state)
        kfold = KFoldCV(k=self.cv)
        self.results_ = []
        t0 = time.time()

        best_score = -np.inf if self.higher_is_better else np.inf
        best_params = None

        for iteration in range(self.n_iter):
            if self.time_budget and (time.time() - t0) >= self.time_budget:
                break

            params = {k: rng.choice(v) for k, v in self.param_distributions.items()}
            # Convert numpy types to Python scalars
            params = {k: v.item() if hasattr(v, "item") else v for k, v in params.items()}
            scores = []

            for X_tr, y_tr, X_val, y_val in kfold.split(X, y):
                model = self.model_class(**params)
                model.fit(X_tr, y_tr)
                s = _score(model, X_val, y_val, self.task, self.scoring)
                scores.append(s)

            avg_score = float(np.mean(scores))
            self.results_.append({"params": params, "score": avg_score, "iteration": iteration})

            is_better = avg_score > best_score if self.higher_is_better else avg_score < best_score
            if is_better:
                best_score = avg_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        return self

    def best_model(self, X: np.ndarray, y: np.ndarray):
        model = self.model_class(**self.best_params_)
        model.fit(X, y)
        return model
