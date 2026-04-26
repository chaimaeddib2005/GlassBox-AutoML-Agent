"""
glassbox.preprocessing.encoders
---------------------------------
One-Hot Encoding (nominal) and Label Encoding (ordinal).
"""

import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None
        self.n_features_in_ = None

    def _to_str(self, col):
        """Convert column to string safely (handles None, numbers, etc.)"""
        return np.array([str(v) for v in col])

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.categories_ = []

        for i in range(self.n_features_in_):
            col = self._to_str(X[:, i])  # 🔥 FIX: normalize type
            cats = np.unique(col)        # now safe
            self.categories_.append(cats)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        cols = []

        for i, cats in enumerate(self.categories_):
            col = self._to_str(X[:, i])  # 🔥 FIX: same normalization

            for cat in cats:
                cols.append((col == cat).astype(float))

        return np.column_stack(cols)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names(self, input_features=None) -> list:
        names = []
        for i, cats in enumerate(self.categories_):
            prefix = input_features[i] if input_features else f"x{i}"
            for cat in cats:
                names.append(f"{prefix}_{cat}")
        return names

# import numpy as np


# class OneHotEncoder:
#     """
#     Convert nominal categorical column(s) into binary indicator columns.

#     Parameters
#     ----------
#     sparse : bool
#         Not used (kept for API familiarity). Always returns dense np.ndarray.
#     """

#     def __init__(self):
#         self.categories_ = None  # list of arrays, one per feature
#         self.n_features_in_ = None

#     def fit(self, X: np.ndarray) -> "OneHotEncoder":
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)
#         self.n_features_in_ = X.shape[1]
#         self.categories_ = []
#         for i in range(self.n_features_in_):
#             cats = np.unique(X[:, i])
#             self.categories_.append(cats)
#         return self

#     def transform(self, X: np.ndarray) -> np.ndarray:
#         if self.categories_ is None:
#             raise RuntimeError("Call fit() before transform().")
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)
#         cols = []
#         for i, cats in enumerate(self.categories_):
#             for cat in cats:
#                 cols.append((X[:, i] == cat).astype(float))
#         return np.column_stack(cols)

#     def fit_transform(self, X: np.ndarray) -> np.ndarray:
#         return self.fit(X).transform(X)

#     def get_feature_names(self, input_features=None) -> list:
#         names = []
#         for i, cats in enumerate(self.categories_):
#             prefix = input_features[i] if input_features else f"x{i}"
#             for cat in cats:
#                 names.append(f"{prefix}_{cat}")
#         return names

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self._class_to_idx = None

    def fit(self, y: np.ndarray) -> "LabelEncoder":
        # Convert everything to string to avoid mixed-type crashes
        y_str = np.array([str(v) for v in y])
        
        self.classes_ = np.unique(y_str)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before transform().")
        
        y_str = np.array([str(v) for v in y])
        
        return np.array([
            self._class_to_idx.get(v, -1)  # safe fallback
            for v in y_str
        ])

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
        int_to_class = {i: c for c, i in self._class_to_idx.items()}
        return np.array([int_to_class.get(int(v), None) for v in y_encoded])

# class LabelEncoder:
#     """
#     Encode ordinal (or any) categorical labels as integers 0 … n_classes-1.
#     Works on a single 1-D array.
#     """

#     def __init__(self):
#         self.classes_ = None
#         self._class_to_idx = None

#     def fit(self, y: np.ndarray) -> "LabelEncoder":
#         self.classes_ = np.unique(y)
#         self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
#         return self

#     def transform(self, y: np.ndarray) -> np.ndarray:
#         if self.classes_ is None:
#             raise RuntimeError("Call fit() before transform().")
#         return np.array([self._class_to_idx[v] for v in y])

#     def fit_transform(self, y: np.ndarray) -> np.ndarray:
#         return self.fit(y).transform(y)

#     def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
#         return np.array([self.classes_[i] for i in y_encoded])
