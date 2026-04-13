"""
glassbox.autofit
-----------------
AutoFit: End-to-end AutoML pipeline.
EDA → Preprocessing → Model Search → Evaluation → JSON Report
"""

import numpy as np
import time
from glassbox.eda.inspector import Inspector
from glassbox.preprocessing.imputer import SimpleImputer
from glassbox.preprocessing.scalers import StandardScaler
from glassbox.preprocessing.encoders import LabelEncoder
from glassbox.models.linear import LinearRegression, LogisticRegression
from glassbox.models.tree import DecisionTree
from glassbox.models.forest import RandomForest
from glassbox.models.naive_bayes import GaussianNaiveBayes
from glassbox.models.knn import KNearestNeighbors
from glassbox.optimization.cross_validation import KFoldCV
from glassbox.evaluation.metrics import ClassificationMetrics, RegressionMetrics


_CLASSIFICATION_MODELS = {
    "logistic_regression": (LogisticRegression, {"lr": 0.1, "epochs": 500}),
    "decision_tree":       (DecisionTree, {"task": "classification", "max_depth": 5}),
    "random_forest":       (RandomForest, {"task": "classification", "n_estimators": 50, "max_depth": 5}),
    "naive_bayes":         (GaussianNaiveBayes, {}),
    "knn":                 (KNearestNeighbors, {"task": "classification", "k": 5}),
}

_REGRESSION_MODELS = {
    "linear_regression": (LinearRegression, {"lr": 0.01, "epochs": 1000}),
    "decision_tree":     (DecisionTree, {"task": "regression", "max_depth": 5}),
    "random_forest":     (RandomForest, {"task": "regression", "n_estimators": 50, "max_depth": 5}),
    "knn":               (KNearestNeighbors, {"task": "regression", "k": 5}),
}


class AutoFit:
    """
    Automated Machine Learning pipeline.

    Parameters
    ----------
    task : str          'classification' | 'regression' | 'auto'
                        'auto' detects from target column cardinality.
    target_col : int    Index of the target column in the input array.
                        Default -1 (last column).
    cv : int            K-Fold splits for evaluation (default 5).
    time_budget : float Max seconds for model search (default 60).
    scale : bool        Apply StandardScaler to features (default True).
    impute : bool       Fill missing values before fitting (default True).
    random_state : int  Seed for reproducibility.
    verbose : bool      Print progress during fit.

    Usage
    -----
    >>> af = AutoFit(task='classification', target_col=-1)
    >>> report = af.fit(data_array, feature_names=["age","income","label"])
    >>> predictions = af.predict(new_X)
    >>> print(af.explain())
    """

    def __init__(
        self,
        task: str = "auto",
        target_col: int = -1,
        cv: int = 5,
        time_budget: float = 60.0,
        scale: bool = True,
        impute: bool = True,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.task = task
        self.target_col = target_col
        self.cv = cv
        self.time_budget = time_budget
        self.scale = scale
        self.impute = impute
        self.random_state = random_state
        self.verbose = verbose

        # Fitted artifacts
        self.task_ = None
        self.inspector_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.label_enc_ = None
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = None
        self.all_scores_ = {}
        self.feature_names_ = None
        self.report_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GlassBox] {msg}")

    def _detect_task(self, y: np.ndarray) -> str:
        unique = np.unique(y)
        if len(unique) <= 20 and (
            y.dtype.kind in ("U", "S", "O") or len(unique) / len(y) < 0.05
        ):
            return "classification"
        return "regression"

    def fit(self, data: np.ndarray, feature_names: list = None) -> dict:
        """
        Run full AutoML pipeline on raw data array.

        Parameters
        ----------
        data : np.ndarray       Shape (n_samples, n_features+1). Target = last col by default.
        feature_names : list    Optional column names (including target).

        Returns
        -------
        dict  JSON-serialisable report.
        """
        t_start = time.time()
        n_cols = data.shape[1]

        # Split features and target
        col_idx = list(range(n_cols))
        tgt = self.target_col if self.target_col >= 0 else n_cols + self.target_col
        feat_idx = [i for i in col_idx if i != tgt]

        X_raw = data[:, feat_idx]
        y_raw = data[:, tgt]

        if feature_names:
            self.feature_names_ = [feature_names[i] for i in feat_idx]
            target_name = feature_names[tgt]
        else:
            self.feature_names_ = [f"feature_{i}" for i in feat_idx]
            target_name = "target"

        self._log(f"Dataset: {X_raw.shape[0]} samples × {X_raw.shape[1]} features  |  target='{target_name}'")

        # ── EDA ──────────────────────────────────────────────────────────────
        self._log("Running EDA …")
        self.inspector_ = Inspector()
        eda_report = self.inspector_.analyze(X_raw, feature_names=self.feature_names_)

        # ── Detect task ───────────────────────────────────────────────────────
        try:
            y_numeric = y_raw.astype(float)
            self.task_ = self._detect_task(y_numeric) if self.task == "auto" else self.task
        except (ValueError, TypeError):
            self.task_ = "classification"

        self._log(f"Task detected: {self.task_}")

        # ── Preprocessing ─────────────────────────────────────────────────────
        X = X_raw.copy()

        if self.impute:
            self._log("Imputing missing values …")
            self.imputer_ = SimpleImputer(strategy="mean")
            X = self.imputer_.fit_transform(X).astype(float)
        else:
            X = X.astype(float)

        if self.scale:
            self._log("Scaling features (StandardScaler) …")
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Encode target
        if self.task_ == "classification":
            self.label_enc_ = LabelEncoder()
            y = self.label_enc_.fit_transform(y_raw).astype(int)
        else:
            y = y_raw.astype(float)

        # ── Model search ──────────────────────────────────────────────────────
        model_zoo = _CLASSIFICATION_MODELS if self.task_ == "classification" else _REGRESSION_MODELS
        kfold = KFoldCV(k=self.cv, random_state=self.random_state)

        def score_fn(yt, yp):
            if self.task_ == "classification":
                return ClassificationMetrics(yt, yp).accuracy()
            else:
                return RegressionMetrics(yt, yp).r2()

        best_score = -np.inf
        best_model = None
        best_name = None
        t_budget_end = time.time() + self.time_budget

        self._log(f"Searching {len(model_zoo)} models with {self.cv}-fold CV …")
        for name, (ModelClass, params) in model_zoo.items():
            if time.time() > t_budget_end:
                self._log("Time budget reached — stopping search.")
                break
            try:
                scores = kfold.cross_val_score(ModelClass, params, X, y, score_fn)
                avg = float(np.mean(scores))
                self.all_scores_[name] = {"mean": round(avg, 4), "std": round(float(np.std(scores)), 4)}
                self._log(f"  {name:<25} CV score = {avg:.4f} ± {np.std(scores):.4f}")
                if avg > best_score:
                    best_score = avg
                    best_name = name
            except Exception as e:
                self._log(f"  {name:<25} FAILED: {e}")

        # Retrain best model on full data
        best_cls, best_params = model_zoo[best_name]
        best_model = best_cls(**best_params)
        best_model.fit(X, y)

        self.best_model_ = best_model
        self.best_model_name_ = best_name
        self.best_score_ = round(best_score, 4)

        elapsed = round(time.time() - t_start, 2)
        self._log(f"Best model: {best_name}  (score={best_score:.4f})  — finished in {elapsed}s")

        # ── Build report ──────────────────────────────────────────────────────
        metric_name = "accuracy" if self.task_ == "classification" else "r2"
        feature_imp = {}
        if hasattr(best_model, "feature_importance"):
            feature_imp = best_model.feature_importance(self.feature_names_)

        self.report_ = {
            "task":        self.task_,
            "target":      target_name,
            "n_samples":   int(X_raw.shape[0]),
            "n_features":  int(X_raw.shape[1]),
            "eda":         eda_report,
            "models_tried": self.all_scores_,
            "best_model":  best_name,
            f"best_{metric_name}": self.best_score_,
            "feature_importance": feature_imp,
            "elapsed_seconds": elapsed,
        }
        return self.report_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted pipeline to new samples."""
        if self.best_model_ is None:
            raise RuntimeError("Call fit() first.")
        X = X.copy()
        if self.imputer_:
            X = self.imputer_.transform(X).astype(float)
        else:
            X = X.astype(float)
        if self.scaler_:
            X = self.scaler_.transform(X)
        preds = self.best_model_.predict(X)
        if self.task_ == "classification" and self.label_enc_:
            preds = self.label_enc_.inverse_transform(preds.astype(int))
        return preds

    def explain(self) -> str:
        """Return a human-readable explanation of the best model's decisions."""
        if self.report_ is None:
            return "No model fitted yet."
        r = self.report_
        lines = [
            "=" * 50,
            "  GlassBox AutoFit — Explainability Report",
            "=" * 50,
            f"Task            : {r['task']}",
            f"Target column   : {r['target']}",
            f"Dataset         : {r['n_samples']} rows × {r['n_features']} features",
            f"Best model      : {r['best_model']}",
        ]
        metric = "best_accuracy" if r["task"] == "classification" else "best_r2"
        lines.append(f"Best CV score   : {r.get(metric, 'N/A')}")
        lines.append(f"Time taken      : {r['elapsed_seconds']}s")
        lines += ["", "All model scores:"]
        for name, s in r["models_tried"].items():
            marker = " ◄ best" if name == r["best_model"] else ""
            lines.append(f"  {name:<25} {s['mean']:.4f} ± {s['std']:.4f}{marker}")
        if r.get("feature_importance"):
            lines += ["", "Feature importance (top 10):"]
            for feat, imp in list(r["feature_importance"].items())[:10]:
                bar = "█" * int(imp * 30)
                lines.append(f"  {feat:<25} {imp:.4f}  {bar}")
        lines.append("=" * 50)
        return "\n".join(lines)
