"""
GlassBox AutoML — IronClaw MCP Tool (FastMCP)
"""
import csv, io, sys
import numpy as np
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, '/home/chaima-eddib/GlassBox-AutoML-Agent')
from glassbox import AutoFit

mcp = FastMCP("glassbox-automl")

@mcp.tool()
def autofit(csv_text: str, target_column: str, task: str = "auto") -> str:
    """
    Run automated machine learning on CSV data.
    Call this when the user wants to build a model,
    predict a column, or analyse which features matter most.
    """
    try:
        reader = csv.DictReader(io.StringIO(csv_text.strip()))
        rows = list(reader)
    except Exception as e:
        return f"CSV parse error: {e}"

    if not rows:
        return "Error: CSV is empty."

    headers = list(rows[0].keys())
    if target_column not in headers:
        return f"Column '{target_column}' not found. Available: {', '.join(headers)}"

    data = np.array([[row[h] for h in headers] for row in rows])
    target_idx = headers.index(target_column)

    try:
        af = AutoFit(task=task, target_col=target_idx, cv=5, verbose=False)
        af.fit(data, feature_names=headers)
        return af.explain()
    except Exception as e:
        return f"AutoFit error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")