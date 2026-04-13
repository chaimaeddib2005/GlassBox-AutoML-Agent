import json, csv, io
import numpy as np
import sys
sys.path.insert(0, '..')          # finds your glassbox/ package

from glassbox import AutoFit

# The agent calls this function when user asks for a model
def autofit(csv_text: str, target_column: str, task: str = "auto") -> str:
    """
    Parameters
    ----------
    csv_text      : the raw CSV the user pasted
    target_column : the column name to predict
    task          : 'classification', 'regression', or 'auto'
    
    Returns
    -------
    A plain-text explanation the agent reads and repeats to the user
    """
    # 1. Parse CSV into numpy array
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    headers = list(rows[0].keys())
    data = [[row[h] for h in headers] for row in rows]
    data = np.array(data)
    
    # 2. Find target column index
    if target_column not in headers:
        return f"Error: column '{target_column}' not found. Available: {headers}"
    target_idx = headers.index(target_column)
    
    # 3. Run your AutoFit pipeline
    af = AutoFit(task=task, target_col=target_idx, verbose=False)
    af.fit(data, feature_names=headers)
    
    # 4. Return the explanation — agent reads this and tells the user
    return af.explain()