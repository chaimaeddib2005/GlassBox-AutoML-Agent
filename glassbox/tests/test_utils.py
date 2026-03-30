"""Tests for glassbox.utils math primitives."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glassbox.utils import (
    mean, median, mode, std, skewness, kurtosis,
    pearson_correlation, pearson_matrix,
    flag_outliers, cap_outliers,
    euclidean_distance, manhattan_distance,
    sigmoid, add_bias,
)


def test_mean():
    assert abs(mean(np.array([1, 2, 3, 4, 5])) - 3.0) < 1e-9

def test_median_odd():
    assert median(np.array([3, 1, 2])) == 2.0

def test_median_even():
    assert median(np.array([1, 2, 3, 4])) == 2.5

def test_mode():
    assert mode(np.array([1, 2, 2, 3])) == 2

def test_std():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert abs(std(x) - 2.0) < 1e-6

def test_skewness_symmetric():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(skewness(x)) < 1e-6

def test_kurtosis_normal():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, 100000)
    assert abs(kurtosis(x)) < 0.1

def test_pearson_perfect():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(pearson_correlation(x, x) - 1.0) < 1e-9

def test_pearson_anti():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(pearson_correlation(x, -x) + 1.0) < 1e-9

def test_pearson_matrix_shape():
    X = np.random.rand(20, 4)
    mat = pearson_matrix(X)
    assert mat.shape == (4, 4)
    assert all(abs(mat[i, i] - 1.0) < 1e-9 for i in range(4))

def test_flag_outliers():
    x = np.array([1.0, 2.0, 2.0, 3.0, 100.0])
    mask = flag_outliers(x)
    assert mask[-1] == True
    assert mask[0] == False

def test_cap_outliers():
    x = np.array([1.0, 2.0, 3.0, 100.0])
    capped = cap_outliers(x)
    assert capped[-1] < 100.0

def test_euclidean():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert abs(euclidean_distance(a, b) - 5.0) < 1e-9

def test_manhattan():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert abs(manhattan_distance(a, b) - 7.0) < 1e-9

def test_sigmoid_bounds():
    x = np.array([-1000.0, 0.0, 1000.0])
    s = sigmoid(x)
    assert s[0] >= 0 and s[2] <= 1
    assert abs(s[1] - 0.5) < 1e-9

def test_add_bias():
    X = np.array([[1, 2], [3, 4]])
    Xb = add_bias(X)
    assert Xb.shape == (2, 3)
    assert np.all(Xb[:, 0] == 1)


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
