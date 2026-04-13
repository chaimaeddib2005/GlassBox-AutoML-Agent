"""Tests for preprocessing modules."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glassbox.preprocessing.imputer import SimpleImputer
from glassbox.preprocessing.scalers import MinMaxScaler, StandardScaler
from glassbox.preprocessing.encoders import OneHotEncoder, LabelEncoder


# ── Imputer ──────────────────────────────────────────────────────────────────

def test_imputer_mean():
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
    imp = SimpleImputer(strategy="mean")
    out = imp.fit_transform(X)
    assert not np.isnan(out.astype(float)).any()
    assert abs(float(out[1, 0]) - 2.0) < 1e-9   # mean of [1,3]

def test_imputer_median():
    X = np.array([[1.0], [2.0], [np.nan], [100.0]])
    imp = SimpleImputer(strategy="median")
    out = imp.fit_transform(X)
    assert float(out[2, 0]) == 2.0  # median of [1,2,100] = 2.0

def test_imputer_mode():
    X = np.array([[1.0], [2.0], [2.0], [np.nan]])
    imp = SimpleImputer(strategy="mode")
    out = imp.fit_transform(X)
    assert float(out[3, 0]) == 2.0


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

def test_minmax_range():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    sc = MinMaxScaler()
    out = sc.fit_transform(X)
    assert abs(out[:, 0].min()) < 1e-9
    assert abs(out[:, 0].max() - 1.0) < 1e-9

def test_minmax_inverse():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    X_back = sc.inverse_transform(X_scaled)
    assert np.allclose(X, X_back)

def test_minmax_constant_col():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    sc = MinMaxScaler()
    out = sc.fit_transform(X)   # should not raise
    assert np.all(out[:, 0] == 0.0)


# ── StandardScaler ────────────────────────────────────────────────────────────

def test_standard_mean_zero():
    X = np.random.rand(50, 3)
    sc = StandardScaler()
    out = sc.fit_transform(X)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9)

def test_standard_std_one():
    X = np.random.rand(50, 3)
    sc = StandardScaler()
    out = sc.fit_transform(X)
    assert np.allclose(out.std(axis=0), 1, atol=1e-6)

def test_standard_inverse():
    X = np.random.rand(20, 2)
    sc = StandardScaler()
    out = sc.fit_transform(X)
    back = sc.inverse_transform(out)
    assert np.allclose(X, back, atol=1e-9)


# ── OneHotEncoder ─────────────────────────────────────────────────────────────

def test_ohe_shape():
    X = np.array([["a"], ["b"], ["c"], ["a"]])
    enc = OneHotEncoder()
    out = enc.fit_transform(X)
    assert out.shape == (4, 3)

def test_ohe_binary():
    X = np.array([["cat"], ["dog"], ["cat"]])
    enc = OneHotEncoder()
    out = enc.fit_transform(X)
    assert out.sum(axis=1).tolist() == [1.0, 1.0, 1.0]

def test_ohe_feature_names():
    X = np.array([["red"], ["blue"]])
    enc = OneHotEncoder()
    enc.fit(X)
    names = enc.get_feature_names(["color"])
    assert "color_red" in names and "color_blue" in names


# ── LabelEncoder ──────────────────────────────────────────────────────────────

def test_label_encode():
    y = np.array(["cat", "dog", "cat", "bird"])
    le = LabelEncoder()
    out = le.fit_transform(y)
    assert len(np.unique(out)) == 3

def test_label_inverse():
    y = np.array(["a", "b", "c", "a"])
    le = LabelEncoder()
    encoded = le.fit_transform(y)
    back = le.inverse_transform(encoded)
    assert np.all(back == y)


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
