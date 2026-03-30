"""
examples/autofit_demo.py
------------------------
Full end-to-end demonstration of GlassBox-AutoML using synthetic data.
Run from the project root:  python examples/autofit_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from glassbox import AutoFit
from glassbox.eda.inspector import Inspector
from glassbox.evaluation.metrics import ClassificationMetrics, RegressionMetrics

print("=" * 60)
print("  GlassBox-AutoML — End-to-End Demo")
print("=" * 60)

# ── 1. Generate synthetic classification dataset ──────────────────────────────
rng = np.random.RandomState(42)
n = 300
age     = rng.randint(18, 70, n).astype(float)
income  = rng.normal(50000, 15000, n)
score   = rng.uniform(0, 100, n)
label   = ((age > 35) & (income > 45000) | (score > 75)).astype(int)

# Introduce some missing values
age[rng.choice(n, 10, replace=False)] = np.nan
income[rng.choice(n, 8, replace=False)] = np.nan

data_clf = np.column_stack([age, income, score, label])
feature_names_clf = ["age", "income", "credit_score", "approved"]

print("\n[1] Classification task — loan approval prediction")
print(f"    Dataset: {data_clf.shape[0]} rows × {data_clf.shape[1]-1} features + target")

af_clf = AutoFit(task="classification", target_col=-1, cv=5, time_budget=60, verbose=True)
report_clf = af_clf.fit(data_clf, feature_names=feature_names_clf)

print()
print(af_clf.explain())

# Quick validation on held-out slice
X_test_clf = data_clf[-50:, :-1]
y_test_clf = data_clf[-50:, -1].astype(int)
preds_clf = af_clf.predict(X_test_clf).astype(int)
cm = ClassificationMetrics(y_test_clf, preds_clf)
print(f"\nHeld-out accuracy : {cm.accuracy():.4f}")


# ── 2. Generate synthetic regression dataset ──────────────────────────────────
print("\n" + "=" * 60)
print("[2] Regression task — house price prediction")

sqft    = rng.uniform(500, 4000, n)
rooms   = rng.randint(1, 7, n).astype(float)
age_h   = rng.uniform(0, 50, n)
price   = 150 * sqft + 10000 * rooms - 500 * age_h + rng.normal(0, 8000, n)

data_reg = np.column_stack([sqft, rooms, age_h, price])
feature_names_reg = ["sqft", "rooms", "house_age", "price"]

print(f"    Dataset: {data_reg.shape[0]} rows × {data_reg.shape[1]-1} features + target")

af_reg = AutoFit(task="regression", target_col=-1, cv=5, time_budget=60, verbose=True)
report_reg = af_reg.fit(data_reg, feature_names=feature_names_reg)

print()
print(af_reg.explain())

X_test_reg = data_reg[-50:, :-1]
y_test_reg = data_reg[-50:, -1]
preds_reg  = af_reg.predict(X_test_reg)
rm = RegressionMetrics(y_test_reg, preds_reg)
print(f"\nHeld-out R²  : {rm.r2():.4f}")
print(f"Held-out MAE : {rm.mae():,.2f}")


# ── 3. Standalone EDA demo ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[3] Standalone EDA")

insp = Inspector()
report = insp.analyze(data_clf[:, :-1], feature_names=["age", "income", "credit_score"])
print(insp.summary())

print("\nDemo complete.")
