
import numpy as np
import sys, os, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from glassbox.autofit import AutoFit

import google.generativeai as genai

TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="autofit_tool",
                description=(
                    "Runs the full GlassBox AutoML pipeline on a dataset. "
                    "Performs EDA, preprocessing, model search with cross-validation, "
                    "and returns a report with the best model, score, and feature importances. "
                    "Use this whenever the user wants to build, train, or fit a model."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "csv_path": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Path to the uploaded CSV file"
                        ),
                        "task": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The ML task type: 'classification', 'regression', or 'auto'"
                        ),
                        "target_column": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Name of the column to predict (e.g. 'Survived', 'Price')"
                        ),
                    },
                    required=["csv_path", "task", "target_column"]
                )
            )
        ]
    )
]

# TOOLS = [
#     {
#         "name": "autofit_tool",
#         "description": (
#             "Runs the full GlassBox AutoML pipeline on a dataset. "
#             "Performs EDA, preprocessing, model search with cross-validation, "
#             "and returns a report with the best model, score, and feature importances. "
#             "Use this whenever the user wants to build, train, or fit a model."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "csv_path": {
#                     "type": "string",
#                     "description": "Path to the uploaded CSV file"
#                 },
#                 "task": {
#                     "type": "string",
#                     "enum": ["classification", "regression", "auto"],
#                     "description": "The ML task type. Use 'auto' if unsure."
#                 },
#                 "target_column": {
#                     "type": "string",
#                     "description": "Name of the column to predict (e.g. 'Approved', 'Price')"
#                 }
#             },
#             "required": ["csv_path", "task", "target_column"]
#         }
#     }
# ]


# When Gemini says "call autofit_tool", this is what actually runs.

def execute_tool(tool_name: str, args: dict) -> dict:
    if tool_name == "autofit_tool":
        return _run_autofit(**args)
    return {"error": f"Unknown tool: {tool_name}"}

# def _run_autofit(csv_path: str, task: str, target_column: str) -> dict:
#     # 1. Load CSV into numpy array
#     with open(csv_path, newline='') as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)

#     feature_names = list(rows[0].keys())

#     if target_column not in feature_names:
#         return {"error": f"Column '{target_column}' not found. Available: {feature_names}"}

#     target_idx = feature_names.index(target_column)

#     # Convert to numpy (all values as strings first, AutoFit handles types)
#     data = []
#     for row in rows:
#         data.append(list(row.values()))
#     data = np.array(data)

#     # 2. Run AutoFit — this calls your actual glassbox library
#     af = AutoFit(task=task, target_col=target_idx, cv=3, time_budget=60, verbose=False)
#     report = af.fit(data, feature_names=feature_names)
#     explanation = af.explain()

#     return {
#         "status": "success",
#         "report": report,
#         "explanation": explanation   # this is the human-readable text
#     }
import pandas as pd 

# def _run_autofit(csv_path: str, task: str, target_column: str) -> dict:
#     df = pd.read_csv(csv_path)

#     if target_column not in df.columns:
#         return {"error": f"Column '{target_column}' not found. Available: {list(df.columns)}"}

#     # ── Replicate exactly what your titanic_demo.py does ──────────────

#     # 1. Drop columns that are pure identifiers / high cardinality text
#     #    (Name, Ticket, Cabin in Titanic — useless for ML anyway)
#     for col in df.columns:
#         if col == target_column:
#             continue
#         if df[col].dtype == object and df[col].nunique() > 10:
#             df = df.drop(columns=[col])

#     # 2. Fill missing values manually before passing to AutoFit
#     #    (because EDA runs before AutoFit's own imputer)
#     for col in df.columns:
#         if df[col].dtype in ['float64', 'int64']:
#             df[col] = df[col].fillna(df[col].median())
#         else:
#             df[col] = df[col].fillna(df[col].mode()[0])

#     # 3. Encode remaining text columns to numbers (Sex → 0/1, Embarked → dummies)
#     text_cols = [c for c in df.columns if c != target_column and df[c].dtype == object]
#     if text_cols:
#         df = pd.get_dummies(df, columns=text_cols, drop_first=True)

#     # 4. Convert booleans to int
#     for col in df.columns:
#         if df[col].dtype == bool:
#             df[col] = df[col].astype(int)

#     # ── Now pass to AutoFit — all numeric, no strings, no NaNs ────────
#     feature_names = list(df.columns)
#     target_idx = feature_names.index(target_column)
#     data = df.values.astype(float)

#     af = AutoFit(task=task, target_col=target_idx, cv=3, time_budget=60, verbose=False)
#     report = af.fit(data, feature_names=feature_names)
#     explanation = af.explain()

#     return {
#         "status": "success",
#         "report": report,
#         "explanation": explanation
#     }

def _run_autofit(csv_path: str, task: str, target_column: str) -> dict:
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Available: {list(df.columns)}"}

    feature_names = list(df.columns)
    target_idx = feature_names.index(target_column)
    data = df.values  # AutoFit now handles everything

    af = AutoFit(task=task, target_col=target_idx, cv=3, time_budget=60, verbose=False)
    report = af.fit(data, feature_names=feature_names)
    explanation = af.explain()

    return {
        "status": "success",
        "report": report,
        "explanation": explanation
    }