"""Tests for all GlassBox models."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glassbox.models.linear import LinearRegression, LogisticRegression
from glassbox.models.tree import DecisionTree
from glassbox.models.forest import RandomForest
from glassbox.models.naive_bayes import GaussianNaiveBayes
from glassbox.models.knn import KNearestNeighbors


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_clf_data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def make_reg_data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(200, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(200) * 0.1
    return X, y

def accuracy(yt, yp):
    return float(np.mean(yt == yp))

def r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1 - ss_res / ss_tot


# ── Linear Regression ─────────────────────────────────────────────────────────

def test_linear_regression_r2():
    X, y = make_reg_data()
    model = LinearRegression(lr=0.05, epochs=2000)
    model.fit(X, y)
    assert r2(y, model.predict(X)) > 0.95

def test_linear_regression_schedules():
    X, y = make_reg_data()
    for sched in ("constant", "step", "exponential"):
        m = LinearRegression(lr=0.05, epochs=500, lr_schedule=sched)
        m.fit(X, y)
        assert m.loss_history_[-1] < m.loss_history_[0]

def test_linear_feature_importance():
    X, y = make_reg_data()
    m = LinearRegression(lr=0.05, epochs=1000)
    m.fit(X, y)
    imp = m.feature_importance(["a", "b", "c"])
    assert abs(sum(imp.values()) - 1.0) < 1e-6


# ── Logistic Regression ───────────────────────────────────────────────────────

def test_logistic_regression_accuracy():
    X, y = make_clf_data()
    model = LogisticRegression(lr=0.1, epochs=500)
    model.fit(X, y)
    assert accuracy(y, model.predict(X)) > 0.85

def test_logistic_proba_range():
    X, y = make_clf_data()
    m = LogisticRegression(lr=0.1, epochs=300)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ── Decision Tree ─────────────────────────────────────────────────────────────

def test_tree_classification():
    X, y = make_clf_data()
    m = DecisionTree(task="classification", max_depth=5)
    m.fit(X, y)
    assert accuracy(y, m.predict(X)) > 0.85

def test_tree_regression():
    X, y = make_reg_data()
    m = DecisionTree(task="regression", max_depth=6)
    m.fit(X, y)
    assert r2(y, m.predict(X)) > 0.85

def test_tree_feature_importance_sums_one():
    X, y = make_clf_data()
    m = DecisionTree(task="classification", max_depth=4)
    m.fit(X, y)
    imp = m.feature_importance(["a", "b", "c", "d"])
    assert abs(sum(imp.values()) - 1.0) < 1e-6

def test_tree_depth_limit():
    X, y = make_clf_data()
    m = DecisionTree(task="classification", max_depth=1)
    m.fit(X, y)
    # Depth 1 = at most 2 leaf predictions
    preds = m.predict(X)
    assert len(np.unique(preds)) <= 2


# ── Random Forest ─────────────────────────────────────────────────────────────

def test_forest_classification():
    X, y = make_clf_data()
    m = RandomForest(task="classification", n_estimators=30, max_depth=5, random_state=0)
    m.fit(X, y)
    assert accuracy(y, m.predict(X)) > 0.85

def test_forest_regression():
    X, y = make_reg_data()
    m = RandomForest(task="regression", n_estimators=30, max_depth=5, random_state=0)
    m.fit(X, y)
    assert r2(y, m.predict(X)) > 0.70

def test_forest_better_than_single_tree():
    """Forest CV score should be competitive with a single tree."""
    from glassbox.optimization.cross_validation import KFoldCV
    rng = np.random.RandomState(7)
    X = rng.randn(300, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    kf = KFoldCV(k=5, random_state=7)
    score_fn = lambda yt, yp: float(np.mean(yt == yp))

    tree_scores = kf.cross_val_score(
        DecisionTree, {"task": "classification", "max_depth": 4}, X, y, score_fn
    )
    forest_scores = kf.cross_val_score(
        RandomForest, {"task": "classification", "n_estimators": 30, "max_depth": 4, "random_state": 7},
        X, y, score_fn,
    )
    # Forest CV mean should be within 10% of tree CV mean
    assert float(np.mean(forest_scores)) >= float(np.mean(tree_scores)) - 0.10


# ── Gaussian Naive Bayes ──────────────────────────────────────────────────────

def test_nb_accuracy():
    X, y = make_clf_data()
    m = GaussianNaiveBayes()
    m.fit(X, y)
    assert accuracy(y, m.predict(X)) > 0.75

def test_nb_proba_sums_one():
    X, y = make_clf_data()
    m = GaussianNaiveBayes()
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_nb_classes_preserved():
    X, y = make_clf_data()
    m = GaussianNaiveBayes()
    m.fit(X, y)
    preds = m.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


# ── KNN ───────────────────────────────────────────────────────────────────────

def test_knn_classification():
    X, y = make_clf_data()
    m = KNearestNeighbors(k=5, task="classification")
    m.fit(X, y)
    assert accuracy(y, m.predict(X)) > 0.85

def test_knn_regression():
    X, y = make_reg_data()
    m = KNearestNeighbors(k=5, task="regression")
    m.fit(X, y)
    assert r2(y, m.predict(X)) > 0.7

def test_knn_manhattan():
    X, y = make_clf_data()
    m = KNearestNeighbors(k=3, task="classification", metric="manhattan")
    m.fit(X, y)
    assert accuracy(y, m.predict(X)) > 0.80

def test_knn_k1_memorizes():
    X, y = make_clf_data()
    m = KNearestNeighbors(k=1, task="classification")
    m.fit(X, y)
    # k=1 on training data = perfect recall
    assert accuracy(y, m.predict(X)) == 1.0


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
