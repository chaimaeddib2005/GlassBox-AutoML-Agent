"""
glassbox.utils.math
-------------------
Pure-NumPy math primitives used across the library.
All functions operate on numpy arrays and return numpy scalars or arrays.
"""

import numpy as np


# ── Descriptive statistics ──────────────────────────────────────────────────

def mean(x: np.ndarray) -> float:
    return np.sum(x) / len(x)


def median(x: np.ndarray) -> float:
    s = np.sort(x)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def mode(x: np.ndarray):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]


def variance(x: np.ndarray, ddof: int = 0) -> float:
    m = mean(x)
    return np.sum((x - m) ** 2) / (len(x) - ddof)


def std(x: np.ndarray, ddof: int = 0) -> float:
    return np.sqrt(variance(x, ddof=ddof))


def skewness(x: np.ndarray) -> float:
    """Fisher's moment coefficient of skewness."""
    n = len(x)
    m = mean(x)
    s = std(x)
    if s == 0:
        return 0.0
    return (np.sum((x - m) ** 3) / n) / (s ** 3)


def kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition, normal = 0)."""
    n = len(x)
    m = mean(x)
    s = std(x)
    if s == 0:
        return 0.0
    return (np.sum((x - m) ** 4) / n) / (s ** 4) - 3.0


# ── Correlation ─────────────────────────────────────────────────────────────

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = mean(x), mean(y)
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


def pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix for a 2-D array (n_samples, n_features)."""
    n_features = X.shape[1]
    mat = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r = pearson_correlation(X[:, i], X[:, j])
            mat[i, j] = r
            mat[j, i] = r
    return mat


# ── Outlier detection ────────────────────────────────────────────────────────

def iqr_bounds(x: np.ndarray, factor: float = 1.5):
    """Return (lower, upper) IQR fences."""
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def flag_outliers(x: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Return boolean mask; True = outlier."""
    lo, hi = iqr_bounds(x, factor)
    return (x < lo) | (x > hi)


def cap_outliers(x: np.ndarray, factor: float = 1.5) -> np.ndarray:
    lo, hi = iqr_bounds(x, factor)
    return np.clip(x, lo, hi)


# ── Distance metrics ─────────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def pairwise_distances(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute distance matrix (n_X, n_Y).
    metric: 'euclidean' | 'manhattan'
    """
    fn = euclidean_distance if metric == "euclidean" else manhattan_distance
    dists = np.zeros((len(X), len(Y)))
    for i, a in enumerate(X):
        for j, b in enumerate(Y):
            dists[i, j] = fn(a, b)
    return dists


# ── Matrix utilities ──────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for bias/intercept."""
    return np.hstack([np.ones((X.shape[0], 1)), X])
