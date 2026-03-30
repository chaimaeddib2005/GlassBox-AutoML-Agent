"""
glassbox.models.linear
-----------------------
Linear Regression and Logistic Regression via Gradient Descent.
Custom learning-rate schedules: constant, step decay, exponential decay.
"""

import numpy as np
from glassbox.utils import sigmoid, add_bias


def _lr_schedule(schedule: str, base_lr: float, epoch: int, decay: float = 0.1, step: int = 10) -> float:
    if schedule == "constant":
        return base_lr
    elif schedule == "step":
        return base_lr * (decay ** (epoch // step))
    elif schedule == "exponential":
        return base_lr * np.exp(-decay * epoch)
    else:
        raise ValueError(f"Unknown lr_schedule: '{schedule}'. Choose constant | step | exponential.")


class LinearRegression:
    """
    Ordinary Least Squares via Gradient Descent.

    Parameters
    ----------
    lr : float            Initial learning rate.
    epochs : int          Number of full passes.
    lr_schedule : str     'constant' | 'step' | 'exponential'
    decay : float         Decay factor for step/exponential schedules.
    step_size : int       Epoch interval for step decay.
    """

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 1000,
        lr_schedule: str = "constant",
        decay: float = 0.1,
        step_size: int = 100,
    ):
        self.lr = lr
        self.epochs = epochs
        self.lr_schedule = lr_schedule
        self.decay = decay
        self.step_size = step_size
        self.weights_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X_b = add_bias(X)
        n = X_b.shape[0]
        self.weights_ = np.zeros(X_b.shape[1])
        self.loss_history_ = []

        for epoch in range(self.epochs):
            lr_t = _lr_schedule(self.lr_schedule, self.lr, epoch, self.decay, self.step_size)
            y_pred = X_b @ self.weights_
            error = y_pred - y
            grad = (2 / n) * (X_b.T @ error)
            self.weights_ -= lr_t * grad
            loss = float(np.mean(error ** 2))
            self.loss_history_.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self.weights_

    def coef_(self):
        return self.weights_[1:] if self.weights_ is not None else None

    def intercept_(self):
        return self.weights_[0] if self.weights_ is not None else None

    def feature_importance(self, feature_names=None) -> dict:
        coefs = np.abs(self.weights_[1:])
        total = coefs.sum() or 1
        names = feature_names or [f"f{i}" for i in range(len(coefs))]
        return dict(sorted(zip(names, (coefs / total).tolist()), key=lambda x: -x[1]))


class LogisticRegression:
    """
    Binary Logistic Regression via Gradient Descent with sigmoid activation.

    Parameters
    ----------
    lr, epochs, lr_schedule, decay, step_size : same as LinearRegression
    threshold : float   Decision boundary for predict() (default 0.5).
    """

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 1000,
        lr_schedule: str = "constant",
        decay: float = 0.1,
        step_size: int = 100,
        threshold: float = 0.5,
    ):
        self.lr = lr
        self.epochs = epochs
        self.lr_schedule = lr_schedule
        self.decay = decay
        self.step_size = step_size
        self.threshold = threshold
        self.weights_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X_b = add_bias(X)
        n = X_b.shape[0]
        self.weights_ = np.zeros(X_b.shape[1])
        self.loss_history_ = []

        for epoch in range(self.epochs):
            lr_t = _lr_schedule(self.lr_schedule, self.lr, epoch, self.decay, self.step_size)
            y_hat = sigmoid(X_b @ self.weights_)
            error = y_hat - y
            grad = (1 / n) * (X_b.T @ error)
            self.weights_ -= lr_t * grad
            # Binary cross-entropy loss
            eps = 1e-15
            loss = -float(np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(add_bias(X) @ self.weights_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def feature_importance(self, feature_names=None) -> dict:
        coefs = np.abs(self.weights_[1:])
        total = coefs.sum() or 1
        names = feature_names or [f"f{i}" for i in range(len(coefs))]
        return dict(sorted(zip(names, (coefs / total).tolist()), key=lambda x: -x[1]))
