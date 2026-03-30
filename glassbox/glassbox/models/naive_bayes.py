"""
glassbox.models.naive_bayes
----------------------------
Gaussian Naive Bayes classifier with Laplace smoothing.
"""

import numpy as np
from glassbox.utils import mean, std


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes — class-wise mean & variance, Laplace smoothing on priors.

    Parameters
    ----------
    var_smoothing : float
        Added to variance to avoid zero-division (like scikit-learn's var_smoothing).
    alpha : float
        Laplace smoothing parameter for class priors.
    """

    def __init__(self, var_smoothing: float = 1e-9, alpha: float = 1.0):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = len(y)

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.class_priors_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[i] = np.array([mean(X_cls[:, j]) for j in range(n_features)])
            self.vars_[i] = np.array([max(std(X_cls[:, j]) ** 2, self.var_smoothing) for j in range(n_features)])
            # Laplace smoothed prior
            self.class_priors_[i] = (len(X_cls) + self.alpha) / (n_samples + self.alpha * n_classes)

        return self

    def _log_likelihood(self, x: np.ndarray, class_idx: int) -> float:
        m = self.means_[class_idx]
        v = self.vars_[class_idx]
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * v) + ((x - m) ** 2) / v)
        return log_prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")
        log_probs = np.zeros((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            for j in range(len(self.classes_)):
                log_probs[i, j] = np.log(self.class_priors_[j]) + self._log_likelihood(x, j)
        # Softmax for proper probabilities
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
