# GlassBox-AutoML
A **transparent, scratch-built** Automated Machine Learning library with a **NumPy-only** math core, designed to be deployed as a secure tool for **IronClaw (NEAR AI)** agents — allowing users to perform complex data science via natural language.

## Features
| Module | Contents |
|--------|----------|
| **EDA / Inspector** | Mean, median, mode, std, skewness, kurtosis, Pearson matrix, IQR outliers, auto-typing |
| **Preprocessing** | SimpleImputer, MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder |
| **Models** | LinearRegression, LogisticRegression, DecisionTree, RandomForest, GaussianNaiveBayes, KNearestNeighbors |
| **Optimization** | GridSearch, RandomSearch, KFoldCV |
| **Evaluation** | ClassificationMetrics (accuracy, precision, recall, F1, confusion matrix), RegressionMetrics (MAE, MSE, RMSE, R²) |
| **AutoFit** | End-to-end pipeline: EDA → Cleaning → Model Search → Explainability report |

---

## Quick Start
```python
from glassbox import AutoFit
import numpy as np

# data = numpy array, last column = target
af = AutoFit(task="classification", target_col=-1, cv=5, time_budget=60)
report = af.fit(data, feature_names=["age", "income", "credit_score", "approved"])
print(af.explain())
predictions = af.predict(new_X)
```

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/chaimaeddib2005/GlassBox-AutoML-Agent
cd GlassBox-AutoML-Agent
```

### Step 2 — Restore the conda environment
An `environment.yml` file is provided so you can recreate the exact environment used in development:
```bash
conda env create -f environment.yml
conda activate glassbox-agent
```

> **Note:** This requires Python 3.11. The environment will **not** work on Python 3.13+ because some dependencies rely on the `pipes` module which was removed in Python 3.13.

### Step 3 — Install GlassBox in editable mode
```bash
pip install -e .
```

### Step 4 — Verify
```bash
python -c "from glassbox import AutoFit; print('GlassBox ready!')"
```

---

## Run the demo
```bash
python examples/autofit_demo.py
```

## Run all tests
```bash
python tests/test_utils.py
python tests/test_preprocessing.py
python tests/test_models.py
python tests/test_optimization_eval.py
```

---

## IronClaw Agent Integration

GlassBox is designed to work as an **MCP (Model Context Protocol) tool** inside [IronClaw](https://github.com/nearai/ironclaw) — a secure, local AI agent framework. Once registered, the IronClaw agent calls your GlassBox pipeline automatically when a user asks data science questions in natural language.

### How it works
```
User: "Build a model to predict survival from this CSV"
        ↓
IronClaw Agent (LLM) understands the request
        ↓
Calls GlassBox via MCP protocol (agent/tool.py)
        ↓
AutoFit pipeline runs: EDA → Preprocessing → Models → Evaluation
        ↓
IronClaw explains results in plain English
```

---

### Prerequisites
- Ubuntu/Linux machine
- Anaconda installed
- A free Gemini API key from https://aistudio.google.com/apikey

---

### PART 1 — Install PostgreSQL + pgvector

```bash
# Install PostgreSQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Start and enable it
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install pgvector extension
sudo apt install -y postgresql-16-pgvector

# Create the IronClaw database
sudo -u postgres createdb ironclaw

# Enable the vector extension
sudo -u postgres psql ironclaw -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

### PART 2 — Install IronClaw

> The official installer script is broken on Linux x86_64 — download the binary directly instead:

```bash
# Download the binary
curl -LO https://github.com/nearai/ironclaw/releases/download/ironclaw-v0.26.0/ironclaw-x86_64-unknown-linux-gnu.tar.gz

# Extract
tar -xzf ironclaw-x86_64-unknown-linux-gnu.tar.gz

# Move binaries to PATH
mkdir -p ~/.cargo/bin
mv ironclaw-x86_64-unknown-linux-gnu/ironclaw ~/.cargo/bin/ironclaw
mv ironclaw-x86_64-unknown-linux-gnu/sandbox_daemon ~/.cargo/bin/sandbox_daemon

# Add to PATH permanently
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
ironclaw --version
```

---

### PART 3 — IronClaw Onboarding

```bash
ironclaw onboard
```

Answer the wizard like this:

| Question | Answer |
|----------|--------|
| Database path | press Enter (use default) |
| Security/keychain | press Enter (use default) |
| Inference provider | choose `gemini` |
| Gemini API key | paste your key |
| Model | `gemini-2.0-flash-lite` (1500 free requests/day) |
| Embeddings | N |
| Tunnel | N |
| Channels | keep only CLI selected |
| Extensions/tools | press Enter (skip) |
| Docker sandbox | N |
| Heartbeat | N |

---

### PART 4 — Configure the Model

```bash
# Set provider and model
ironclaw models set-provider gemini
ironclaw config set selected_model gemini-2.0-flash-lite
ironclaw config set GEMINI_API_KEY your_key_here

# Disable unused channels to avoid startup errors
ironclaw config set channels.http_enabled false
ironclaw config set channels.gateway_enabled false

# Verify
ironclaw config list | grep -i "model\|provider"
```

---

### PART 5 — Create the MCP Tool Files

The `agent/` folder is already included in this repository. You only need to update the paths to match your machine.

**Edit `agent/tool.py`** — update the `sys.path.insert` line:
```python
sys.path.insert(0, '/home/YOUR_USERNAME/GlassBox-AutoML-Agent')
```

**Edit `agent/run_tool.sh`** — update the username:
```bash
#!/bin/bash
source /home/YOUR_USERNAME/anaconda3/etc/profile.d/conda.sh
conda activate glassbox-agent
exec python3 /home/YOUR_USERNAME/GlassBox-AutoML-Agent/agent/tool.py
```

Make it executable:
```bash
chmod +x agent/run_tool.sh
```

Test it runs without errors:
```bash
conda activate glassbox-agent
python3 agent/tool.py
# Should be silent — press Ctrl+C to stop
```

---

### PART 6 — Register GlassBox as an MCP Tool

```bash
# Replace YOUR_USERNAME with your actual Linux username (run: whoami)
ironclaw mcp add glassbox-automl \
  --transport stdio \
  --command "/home/YOUR_USERNAME/GlassBox-AutoML-Agent/agent/run_tool.sh"

# Verify it was registered
ironclaw mcp list
# Should show: ● glassbox-automl
```

---

### PART 7 — Run IronClaw

```bash
ironclaw run --cli-only
```

At startup you should see:
```
model    gemini-2.0-flash-lite  via gemini
features db:libsql  tools:XX  skills
```

And in the tools panel on the right:
```
glassbox: automl_autofit
```

---

### PART 8 — Test the Integration

Inside the IronClaw chat, paste this:

```
Here is my Titanic data:

PassengerId,Pclass,Name,Sex,Age,Survived
1,3,Braund Mr. Owen Harris,male,22,0
2,1,Cumings Mrs. John Bradley,female,38,1
3,3,Heikkinen Miss. Laina,female,26,1
4,1,Futrelle Mrs. Jacques Heath,female,35,1
5,3,Allen Mr. William Henry,male,35,0
6,3,Moran Mr. James,male,27,0
7,1,McCarthy Mr. Timothy J,male,54,0
8,3,Palsson Master. Gosta Leonard,male,2,0

Predict whether passengers survived.
```

IronClaw will automatically call the `autofit` tool and explain the results in plain English.

---

### Agent Folder Structure
```
agent/
├── tool.py        # FastMCP server — wraps GlassBox AutoFit as a callable tool
└── run_tool.sh    # Shell script — activates conda env and runs tool.py
```

### MCP Tool Input Schema
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `csv_text` | string | yes | Full CSV content as text |
| `target_column` | string | yes | Name of the column to predict |
| `task` | string | no | `classification`, `regression`, or `auto` (default) |

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ironclaw: command not found` | `source ~/.bashrc` |
| `Another instance already running` | `rm ~/.ironclaw/ironclaw.pid` |
| MCP tool not showing in tools list | Re-run `ironclaw mcp add ...` command |
| Rate limit 429 error | Switch key or wait for quota reset (per day) |
| `tool.py` crashes on import | Check `sys.path.insert` path matches your machine |
| HTTP channel error on startup | `ironclaw config set channels.http_enabled false` |
| Empty response from model | Switch to `gemini-2.0-flash-lite` |

---

### Quick Reference

```bash
# Activate environment
conda activate glassbox-agent

# Start IronClaw
ironclaw run --cli-only

# Check registered MCP tools
ironclaw mcp list

# Check current model
ironclaw models status

# Run full diagnostics
ironclaw doctor

# Fix stale PID error
rm ~/.ironclaw/ironclaw.pid
```

---

## Architecture
```
glassbox/
├── autofit.py              # End-to-end AutoML orchestrator
├── eda/
│   └── inspector.py        # EDA: statistics, correlation, outliers, auto-typing
├── preprocessing/
│   ├── imputer.py          # SimpleImputer (mean/median/mode)
│   ├── scalers.py          # MinMaxScaler, StandardScaler
│   └── encoders.py         # OneHotEncoder, LabelEncoder
├── models/
│   ├── linear.py           # LinearRegression, LogisticRegression (gradient descent)
│   ├── tree.py             # DecisionTree (Gini / MSE)
│   ├── forest.py           # RandomForest (bagging + √features)
│   ├── naive_bayes.py      # GaussianNaiveBayes (Laplace smoothing)
│   └── knn.py              # KNearestNeighbors (Euclidean + Manhattan)
├── optimization/
│   ├── search.py           # GridSearch, RandomSearch
│   └── cross_validation.py # KFoldCV
└── evaluation/
    └── metrics.py          # ClassificationMetrics, RegressionMetrics

agent/
├── tool.py                 # MCP server for IronClaw integration
└── run_tool.sh             # Environment activation script
```

---

## Design Principles
- **Zero heavy dependencies** — only NumPy for all math
- **White-box** — every model can explain its decisions
- **WASM-ready** — no C extensions, pure Python + NumPy
- **Modular** — every transformer implements `fit()`, `transform()`, `fit_transform()`
- **Agent-ready** — exposes a clean MCP interface for IronClaw integration

---

## PyPI
```bash
pip install glassbox-automl
```
Package: https://pypi.org/project/glassbox-automl/1.0.0/

## GitHub
https://github.com/chaimaeddib2005/GlassBox-AutoML-Agent

## License
MIT