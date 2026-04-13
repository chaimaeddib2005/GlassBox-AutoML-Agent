import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from glassbox import AutoFit
from glassbox.eda.inspector import Inspector
from glassbox.evaluation.metrics import ClassificationMetrics

print("=" * 60)
print("  GlassBox-AutoML — Titanic Dataset")
print("=" * 60)

# ── 1. Load Titanic dataset ──────────────────────────────────────────────────
df = pd.read_csv("examples/titanic.csv")

print("\n[1] Classification task — survival prediction")

# ── 2. Basic preprocessing ───────────────────────────────────────────────────
# Keep relevant features
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Convert to numpy
data = df.values
feature_names = list(df.columns)

print(f"    Dataset: {data.shape[0]} rows × {data.shape[1]-1} features + target")

# ── 3. Train AutoFit ─────────────────────────────────────────────────────────
af = AutoFit(task="classification", target_col=-1, cv=5, time_budget=60, verbose=True)
report = af.fit(data, feature_names=feature_names)

print()
print(af.explain())

# ── 4. Evaluation ────────────────────────────────────────────────────────────
X_test = data[-100:, :-1]
y_test = data[-100:, -1].astype(int)

preds = af.predict(X_test).astype(int)

cm = ClassificationMetrics(y_test, preds)
print(f"\nHeld-out accuracy : {cm.accuracy():.4f}")

# ── 5. Standalone EDA ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[2] Standalone EDA")

insp = Inspector()
report = insp.analyze(data[:, :-1], feature_names=feature_names[:-1])
print(insp.summary())

print("\nDemo complete.")
