"""
glassbox.evaluation.metrics
-----------------------------
Classification and Regression evaluation metrics — pure NumPy.
"""

import numpy as np


class ClassificationMetrics:
    """
    Compute classification metrics from true and predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Methods
    -------
    accuracy(), precision(), recall(), f1(), confusion_matrix(), report()
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.classes_ = np.unique(np.concatenate([self.y_true, self.y_pred]))

    def accuracy(self) -> float:
        return float(np.mean(self.y_true == self.y_pred))

    def confusion_matrix(self) -> np.ndarray:
        n = len(self.classes_)
        cm = np.zeros((n, n), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for t, p in zip(self.y_true, self.y_pred):
            cm[class_to_idx[t], class_to_idx[p]] += 1
        return cm

    def precision(self, average: str = "macro") -> float:
        cm = self.confusion_matrix()
        per_class = []
        for i in range(len(self.classes_)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            denom = tp + fp
            per_class.append(tp / denom if denom > 0 else 0.0)
        if average == "macro":
            return float(np.mean(per_class))
        counts = np.array([(self.y_true == c).sum() for c in self.classes_])
        return float(np.sum(np.array(per_class) * counts) / counts.sum())

    def recall(self, average: str = "macro") -> float:
        cm = self.confusion_matrix()
        per_class = []
        for i in range(len(self.classes_)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            denom = tp + fn
            per_class.append(tp / denom if denom > 0 else 0.0)
        if average == "macro":
            return float(np.mean(per_class))
        counts = np.array([(self.y_true == c).sum() for c in self.classes_])
        return float(np.sum(np.array(per_class) * counts) / counts.sum())

    def f1(self, average: str = "macro") -> float:
        p = self.precision(average)
        r = self.recall(average)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def report(self) -> str:
        lines = [
            f"{'Metric':<20} {'Score':>8}",
            "-" * 30,
            f"{'Accuracy':<20} {self.accuracy():>8.4f}",
            f"{'Precision (macro)':<20} {self.precision():>8.4f}",
            f"{'Recall (macro)':<20} {self.recall():>8.4f}",
            f"{'F1 (macro)':<20} {self.f1():>8.4f}",
            "",
            "Confusion matrix (rows=true, cols=pred):",
            f"Classes: {list(self.classes_)}",
        ]
        cm = self.confusion_matrix()
        for row in cm:
            lines.append("  " + "  ".join(f"{v:5d}" for v in row))
        return "\n".join(lines)


class RegressionMetrics:
    """
    Compute regression metrics.

    Methods
    -------
    mae(), mse(), rmse(), r2(), report()
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

    def mae(self) -> float:
        return float(np.mean(np.abs(self.y_true - self.y_pred)))

    def mse(self) -> float:
        return float(np.mean((self.y_true - self.y_pred) ** 2))

    def rmse(self) -> float:
        return float(np.sqrt(self.mse()))

    def r2(self) -> float:
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def report(self) -> str:
        return (
            f"{'Metric':<20} {'Score':>12}\n"
            + "-" * 34 + "\n"
            + f"{'MAE':<20} {self.mae():>12.4f}\n"
            + f"{'MSE':<20} {self.mse():>12.4f}\n"
            + f"{'RMSE':<20} {self.rmse():>12.4f}\n"
            + f"{'R²':<20} {self.r2():>12.4f}"
        )
