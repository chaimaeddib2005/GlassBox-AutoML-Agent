GlassBox_AutoML_Agent
lassbox/
│
├── core/
│   └── stats.py                  # Manual implementations: mean, median, mode,
│                                 # std, skewness, kurtosis, Pearson matrix
│
├── eda/
│   └── inspector.py              # Auto-typing, statistical profiling,
│                                 # IQR outlier detection, collinearity flags
│
├── preprocessing/
│   ├── base.py                   # BaseTransformer (fit / transform / fit_transform)
│   ├── imputer.py                # SimpleImputer — mean/median/mode fill
│   ├── scaler.py                 # MinMaxScaler, StandardScaler
│   ├── encoder.py                # OneHotEncoder, LabelEncoder
│   └── pipeline.py               # Pipeline — chains transformers in order
│
├── models/
│   ├── base_model.py             # BaseModel (fit / predict / score)
│   ├── linear.py                 # LinearRegression, LogisticRegression (gradient descent)
│   ├── tree.py                   # DecisionTree — Gini impurity / MSE variance reduction
│   ├── forest.py                 # RandomForest — bagging + feature subspace sampling
│   ├── naive_bayes.py            # GaussianNB — class-wise mean/variance + Laplace smoothing
│   └── knn.py                    # KNN — Euclidean / Manhattan, lazy inference
│
├── evaluation/
│   ├── classification.py         # Accuracy, Precision, Recall, F1, Confusion Matrix
│   └── regression.py             # MAE, MSE, RMSE, R² Score
│
├── optimization/
│   ├── cross_validation.py       # K-Fold splitter
│   ├── grid_search.py            # Exhaustive hyperparameter search
│   └── random_search.py          # Stochastic search with time budget
│
├── agent/
│   ├── autofit.py                # Top-level orchestrator: EDA → clean → search → report
│   ├── tool_wrapper.py           # MCP/IronClaw JSON tool interface
│   └── report.py                 # JSON report builder
│
├── deploy/
│   ├── build_wasm.sh             # Pyodide/MicroPython compile script
│   └── sandbox_test.js           # Smoke-test in Node WASM runtime
│
├── tests/
│   ├── test_stats.py
│   ├── test_scalers.py
│   ├── test_preprocessing.py
│   ├── test_encoders.py
│   ├── test_pipeline.py
│   ├── test_models_simple.py
│   ├── test_models_trees.py
│   ├── test_optimization.py
│   └── test_evaluation.py
│
├── benchmarks/
│   ├── accuracy_vs_sklearn.py    # Validates ≥90% of Scikit-Learn accuracy
│   └── preprocessing_smoke.py
│
├── pyproject.toml                # numpy only as dependency
└── README.md