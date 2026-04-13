"""
glassbox.eda.inspector
----------------------
Non-destructive audit of raw data.
Returns a structured report dict that can be passed to an agent.
"""

import numpy as np
from glassbox.utils import (
    mean, median, mode, std, skewness, kurtosis,
    pearson_matrix, flag_outliers,
)


def _detect_type(col: np.ndarray) -> str:
    """Auto-type a column: 'boolean', 'categorical', or 'numerical'."""
    unique = np.unique(col[~_is_nan(col)])
    if set(unique).issubset({0, 1, True, False, "0", "1", "true", "false", "True", "False"}):
        return "boolean"
    try:
        col.astype(float)
        if len(unique) <= 10 and len(unique) / len(col) < 0.05:
            return "categorical"
        return "numerical"
    except (ValueError, TypeError):
        return "categorical"


def _is_nan(col: np.ndarray) -> np.ndarray:
    try:
        return np.isnan(col.astype(float))
    except (ValueError, TypeError):
        return np.array([v is None or v == "" for v in col])


class Inspector:
    """
    Automated Exploratory Data Analysis.

    Parameters
    ----------
    outlier_factor : float
        IQR multiplier for outlier detection (default 1.5).

    Usage
    -----
    >>> insp = Inspector()
    >>> report = insp.analyze(X, feature_names=["age", "income", "label"])
    >>> print(report)
    """

    def __init__(self, outlier_factor: float = 1.5):
        self.outlier_factor = outlier_factor
        self.report_ = None

    def analyze(self, X: np.ndarray, feature_names: list = None) -> dict:
        """
        Run full EDA on array X (n_samples, n_features).

        Returns
        -------
        dict with keys:
            shape, missing, dtypes, stats, outliers, correlation
        """
        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        report = {
            "shape": {"n_samples": n_samples, "n_features": n_features},
            "missing": {},
            "dtypes": {},
            "stats": {},
            "outliers": {},
            "correlation": None,
        }

        numerical_cols = []
        numerical_idx = []

        for i, name in enumerate(feature_names):
            col = X[:, i]
            nan_mask = _is_nan(col)
            n_missing = int(nan_mask.sum())
            report["missing"][name] = {"count": n_missing, "pct": round(n_missing / n_samples * 100, 2)}

            dtype = _detect_type(col)
            report["dtypes"][name] = dtype

            if dtype == "numerical":
                clean = col[~nan_mask].astype(float)
                if len(clean) == 0:
                    continue
                report["stats"][name] = {
                    "mean":     round(float(mean(clean)), 4),
                    "median":   round(float(median(clean)), 4),
                    "mode":     round(float(mode(clean)), 4),
                    "std":      round(float(std(clean)), 4),
                    "min":      round(float(clean.min()), 4),
                    "max":      round(float(clean.max()), 4),
                    "skewness": round(float(skewness(clean)), 4),
                    "kurtosis": round(float(kurtosis(clean)), 4),
                }
                outlier_mask = flag_outliers(clean, self.outlier_factor)
                report["outliers"][name] = {
                    "count": int(outlier_mask.sum()),
                    "pct":   round(float(outlier_mask.sum()) / len(clean) * 100, 2),
                }
                numerical_cols.append(clean)
                numerical_idx.append(i)
            else:
                unique_vals, counts = np.unique(col[~nan_mask], return_counts=True)
                top_idx = np.argmax(counts)
                report["stats"][name] = {
                    "n_unique":   int(len(unique_vals)),
                    "top_value":  str(unique_vals[top_idx]),
                    "top_freq":   int(counts[top_idx]),
                }

        # Pearson correlation matrix (numerical only)
        if len(numerical_cols) >= 2:
            min_len = min(len(c) for c in numerical_cols)
            mat_data = np.column_stack([c[:min_len] for c in numerical_cols])
            corr = pearson_matrix(mat_data)
            num_names = [feature_names[i] for i in numerical_idx]
            report["correlation"] = {
                "features": num_names,
                "matrix":   corr.tolist(),
            }

        self.report_ = report
        return report

    def summary(self) -> str:
        """Human-readable summary of the last analyze() call."""
        if self.report_ is None:
            return "No analysis run yet. Call analyze() first."
        r = self.report_
        lines = [
            f"Dataset shape : {r['shape']['n_samples']} rows × {r['shape']['n_features']} columns",
            "",
            "Column types :",
        ]
        for name, dtype in r["dtypes"].items():
            missing = r["missing"][name]
            lines.append(f"  {name:30s} {dtype:12s}  missing: {missing['count']} ({missing['pct']}%)")

        lines += ["", "Numerical stats :"]
        for name, s in r["stats"].items():
            if "mean" in s:
                lines.append(
                    f"  {name:30s} mean={s['mean']:.3f}  std={s['std']:.3f}"
                    f"  skew={s['skewness']:.2f}  kurt={s['kurtosis']:.2f}"
                )

        lines += ["", "Outliers (IQR) :"]
        for name, o in r["outliers"].items():
            lines.append(f"  {name:30s} {o['count']} ({o['pct']}%)")

        return "\n".join(lines)
