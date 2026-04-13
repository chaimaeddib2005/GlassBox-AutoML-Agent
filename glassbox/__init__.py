"""
GlassBox-AutoML
A transparent, scratch-built Automated Machine Learning library (NumPy-only core).
"""

from glassbox.autofit import AutoFit
from glassbox.eda.inspector import Inspector
from glassbox.preprocessing.imputer import SimpleImputer
from glassbox.preprocessing.scalers import MinMaxScaler, StandardScaler
from glassbox.preprocessing.encoders import OneHotEncoder, LabelEncoder
from glassbox.models.linear import LinearRegression, LogisticRegression
from glassbox.models.tree import DecisionTree
from glassbox.models.forest import RandomForest
from glassbox.models.naive_bayes import GaussianNaiveBayes
from glassbox.models.knn import KNearestNeighbors
from glassbox.optimization.search import GridSearch, RandomSearch
from glassbox.optimization.cross_validation import KFoldCV
from glassbox.evaluation.metrics import ClassificationMetrics, RegressionMetrics

__version__ = "1.0.0"
__all__ = [
    "AutoFit",
    "Inspector",
    "SimpleImputer",
    "MinMaxScaler", "StandardScaler",
    "OneHotEncoder", "LabelEncoder",
    "LinearRegression", "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "GaussianNaiveBayes",
    "KNearestNeighbors",
    "GridSearch", "RandomSearch",
    "KFoldCV",
    "ClassificationMetrics", "RegressionMetrics",
]
