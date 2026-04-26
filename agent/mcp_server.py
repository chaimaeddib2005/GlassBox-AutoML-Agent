
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


# When Gemini says "call autofit_tool", this is what actually runs.

def execute_tool(tool_name: str, args: dict) -> dict:
    if tool_name == "autofit_tool":
        return _run_autofit(**args)
    return {"error": f"Unknown tool: {tool_name}"}


import pandas as pd 


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