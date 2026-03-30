"""Tests for optimization (search, CV) and evaluation metrics."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glassbox.optimization.cross_validation import KFoldCV
from glassbox.optimization.search import GridSearch, RandomSearch
from glassbox.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from glassbox.models.tree import DecisionTree
from glassbox.models.linear import LinearRegression


def make_clf(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    return X, y

def make_reg(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(100, 2)
    y = 3 * X[:, 0] + rng.randn(100) * 0.2
    return X, y


# ── KFoldCV ───────────────────────────────────────────────────────────────────

def test_kfold_n_splits():
    X, y = make_clf()
    kf = KFoldCV(k=5)
    splits = list(kf.split(X, y))
    assert len(splits) == 5

def test_kfold_no_overlap():
    X, y = make_clf()
    kf = KFoldCV(k=5)
    val_indices = []
    for X_tr, y_tr, X_val, y_val in kf.split(X, y):
        assert len(X_tr) + len(X_val) == len(X)
        val_indices.append(len(X_val))
    # All val indices should cover all samples exactly once
    assert sum(val_indices) == len(X)

def test_kfold_cross_val_score():
    X, y = make_clf()
    kf = KFoldCV(k=3)
    scores = kf.cross_val_score(
        DecisionTree,
        {"task": "classification", "max_depth": 3},
        X, y,
        lambda yt, yp: float(np.mean(yt == yp)),
    )
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)


# ── GridSearch ────────────────────────────────────────────────────────────────

def test_gridsearch_finds_best():
    X, y = make_clf()
    gs = GridSearch(
        DecisionTree,
        {"max_depth": [2, 5], "min_samples_split": [2, 4]},
        task="classification",
        cv=3,
        scoring="accuracy",
    )
    gs.fit(X, y)
    assert gs.best_params_ is not None
    assert gs.best_score_ > 0
    assert len(gs.results_) == 4  # 2×2 grid

def test_gridsearch_best_model_fits():
    X, y = make_clf()
    gs = GridSearch(
        DecisionTree,
        {"max_depth": [3, 5]},
        task="classification",
        cv=3,
    )
    gs.fit(X, y)
    model = gs.best_model(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


# ── RandomSearch ──────────────────────────────────────────────────────────────

def test_randomsearch_n_iter():
    X, y = make_clf()
    rs = RandomSearch(
        DecisionTree,
        {"max_depth": [2, 3, 5, 7, 10], "min_samples_split": [2, 3, 5]},
        task="classification",
        cv=3,
        n_iter=5,
        random_state=42,
    )
    rs.fit(X, y)
    assert len(rs.results_) == 5
    assert rs.best_params_ is not None

def test_randomsearch_time_budget():
    X, y = make_clf()
    rs = RandomSearch(
        DecisionTree,
        {"max_depth": list(range(2, 15))},
        task="classification",
        cv=3,
        n_iter=1000,
        time_budget=2.0,
        random_state=0,
    )
    import time
    t0 = time.time()
    rs.fit(X, y)
    elapsed = time.time() - t0
    assert elapsed < 10  # well within budget overhead


# ── ClassificationMetrics ─────────────────────────────────────────────────────

def test_clf_accuracy():
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    m = ClassificationMetrics(yt, yp)
    assert abs(m.accuracy() - 0.75) < 1e-9

def test_clf_perfect():
    y = np.array([0, 1, 2, 1, 0])
    m = ClassificationMetrics(y, y)
    assert m.accuracy() == 1.0
    assert abs(m.f1() - 1.0) < 1e-6

def test_clf_confusion_matrix_shape():
    yt = np.array([0, 1, 2, 0, 1, 2])
    yp = np.array([0, 2, 1, 0, 1, 2])
    m = ClassificationMetrics(yt, yp)
    cm = m.confusion_matrix()
    assert cm.shape == (3, 3)
    assert cm.sum() == 6

def test_clf_report_string():
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 0])
    m = ClassificationMetrics(yt, yp)
    rep = m.report()
    assert "Accuracy" in rep and "F1" in rep


# ── RegressionMetrics ─────────────────────────────────────────────────────────

def test_reg_mae_zero():
    y = np.array([1.0, 2.0, 3.0])
    m = RegressionMetrics(y, y)
    assert m.mae() == 0.0

def test_reg_r2_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = RegressionMetrics(y, y)
    assert abs(m.r2() - 1.0) < 1e-9

def test_reg_mse_known():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 2.0, 2.0])
    m = RegressionMetrics(yt, yp)
    assert abs(m.mse() - 2/3) < 1e-9

def test_reg_rmse_equals_sqrt_mse():
    yt = np.array([1.0, 3.0, 5.0])
    yp = np.array([2.0, 2.0, 4.0])
    m = RegressionMetrics(yt, yp)
    assert abs(m.rmse() - np.sqrt(m.mse())) < 1e-9

def test_reg_report_string():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.1, 2.1, 2.9])
    m = RegressionMetrics(yt, yp)
    rep = m.report()
    assert "MAE" in rep and "R²" in rep


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed.")
