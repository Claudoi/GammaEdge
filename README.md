# 🧠 GammaEdge — Portfolio Backtesting, Risk Modeling & Reporting Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=Open%20Source%20Initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/badge/Last_Commit-Recently-brightgreen?logo=git&logoColor=white)](https://github.com/Claudoi/GammaEdge/commits/main)
[![Streamlit](https://img.shields.io/badge/Live%20App:%20Open%20in%20Streamlit-App-red?logo=streamlit&logoColor=white)](https://gammaedge.streamlit.app/)


**GammaEdge** is a full-stack platform for **portfolio construction**,  
**backtesting**, **risk modeling**, **scenario analysis**, and **automated reporting**.

The goal of the project is to unify all my quantitative finance tooling into a
production-grade framework that is mathematically rigorous and visually intuitive.

---

GammaEdge is a modular research framework for **portfolio analytics**, designed to support:

- portfolio construction,
- backtesting and turnover analysis,
- risk & factor models,
- Brinson/Euler attribution,
- scenario stress testing,
- and automated **PDF/HTML** reporting.

It is built on a fast NumPy/Polars backend and wrapped in a modern Streamlit interface.

---


## 🚀 Key Features

### 🔹 Backtesting Engine
- Polars-accelerated computations  
- Allocation-agnostic design  
- Full **turnover accounting**  
- Transaction-cost simulation  
- Rebalance diagnostics  

### 🔹 Portfolio Optimization
- **Equal Weight**
- **Risk Parity**
- **HRP**
- **Minimum Variance (PGD)**
- **Tracking-Error Minimization**
- **Robust optimizers** with turnover/cardinality constraints

### 🔹 Scenario Analysis
- Mean & covariance shocks  
- Historical replay windows  
- Tornado sensitivity  
- Crisis crash simulations  
- Beta-correlated market stress  

### 🔹 Performance Attribution
- **Brinson (allocation / selection / interaction)**  
- **Euler risk contributions** (sum to portfolio volatility)  
- Full integration with backtest reporting  

### 🔹 Automated Reporting (NEW)
- Export **PDF** and **HTML** reports  
- Includes:
  - Equity curve  
  - Drawdown  
  - KPIs  
  - Turnover  
  - Attribution tables  
  - Plotly charts  

### 🔹 Streamlit UI (NEW)
- Modern, clean **Home Page**
- Multi-page app (Data → Model → Optimizer → Backtest → Attribution → Reporting)
- Dynamic Plotly interactivity


---


## 🧩 Folder Structure

```bash
GammaEdge/
│
├── README.md
├── pyproject.toml              # Build & dependency configuration (Poetry)
├── pyproject.toml.bak
├── poetry.lock
├── requirements.txt            # Optional pip-based installation
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml     # Ruff, Black, nbstripout, etc.
├── .coveragerc
├── PR_CHECKLIST.md
├── CHANGELOG.md
├── mypy_report.txt
├── content
│
├── .github/                    # CI/CD workflows (GitHub Actions)
│   └── workflows/
│       ├── ci.yml
│       ├── run-ci.yml
│       ├── manual-ci.yml
│       └── hello.yml
│
├── api/                        # Optional FastAPI backend
│   ├── main.py
│   └── routes/
│       ├── backtest.py
│       ├── health.py
│       └── optimize.py
│
├── app/                        # Streamlit frontend
│   ├── Home.py
│   ├── utils.py
│   └── pages/
│       ├── 01_Data.py
│       ├── 02_RiskModel.py
│       ├── 03_Optimizer.py
│       ├── 04_Backtest.py
│       ├── 05_Attribution.py
│       ├── 06_Reporting.py
│       └── 07_Scenarios.py
│
├── configs/                    # Example optimizer configs
│   ├── example_blacklitterman.yaml
│   └── example_markowitz.yaml
│
├── docs/                       # Documentation files
│   ├── index.md
│   ├── risk_models.md
│   ├── api_reference.md
│   ├── optimizers.md
│   └── backtest_attribution.md 
│
├── notebooks/                  # Research notebooks
│   ├── 01_eda_universe.ipynb
│   ├── 02_markowitz_demo.ipynb
│   ├── 03_blacklitterman_views.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── docker/                     # Container setup
│   ├── Dockerfile
│   ├── api.Dockerfile
│   └── docker-compose.yml
│
├── examples/                   
│   └── attribution_usage.py
│
├── portfolio/                  # Core analytical backend
│   ├── __init__.py
│   ├── attribution/            # Attribution engine (Brinson & Euler)
│   │   ├── __init__.py
│   │   ├── brinson.py          # Brinson-style attribution helpers
│   │   ├── euler.py            # Euler risk contributions
│   │   ├── engine.py           # Portfolio-level contributions engine
│   │   └── factor_decomposition.py  
│   ├── backtest/               # Backtest & reporting
│   │   ├── __init__.py
│   │   ├── engine.py           # Core backtest engine
│   │   ├── reporting.py        # Plotly-based reporting (HTML/PDF)
│   │   ├── attribution.py      # Return/weight alignment & contributions
│   │   ├── attribution_reporting.py  # Brinson reporting helpers
│   │   ├── brinson_utils.py    # Low-level attribution utilities
│   │   ├── allocators.py       
│   │   ├── kpis.py
│   │   ├── metrics.py
│   │   └── scenarios.py
│   ├── features/               # Feature engineering & risk models
│   │   ├── __init__.py
│   │   ├── factors.py
│   │   ├── returns.py
│   │   ├── risk_models.py
│   │   └── scenarios.py
│   ├── io/                     # IO & caching helpers
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   └── data_loader.py
│   ├── core/                   # Core utilities and guards
│   │   ├── __init__.py
│   │   ├── __innit__.py        # (legacy / typo shim)
│   │   ├── opt_helpers.py
│   │   ├── utils.py
│   │   ├── logger.py
│   │   ├── compat.py
│   │   ├── guards.py
│   │   └── metrics.py
│   ├── viz/                    # Visualization helpers
│   │   └── plot_utils.py
│   └── optim/                  # Optimizers
│       ├── __init__.py
│       ├── black_litterman.py
│       ├── cardinality.py
│       ├── costs_turnover.py
│       ├── cvar.py
│       ├── exposures.py
│       ├── hrp.py
│       ├── mean_variance.py
│       ├── robust.py
│       ├── te.py
│       └── risk_parity.py
│
├── tools/
│   └── export_report.py        # CLI helper to export HTML/PDF reports
│
├── results/                    # Example backtest outputs
│   ├── bt.json
│   ├── returns_wide.csv
│   ├── group_map.json
│   └── metrics.csv
│
├── reports/                    # Generated reports (examples)
│   ├── backtest_report.html
│   └── backtest_report.pdf
│
└── tests/                      # Test suite
    ├── conftest.py
    ├── utils_dates.py
    ├── test_backtest.py
    ├── test_brinson_vectorized.py
    ├── test_core_guards.py
    ├── test_core_utils.py
    ├── test_data_pipeline.py
    ├── test_export_report_cli.py
    ├── test_frontier_sanity.py
    ├── test_kpis_property.py
    ├── test_mean_variance.py
    ├── test_optimizer_fallbacks.py
    ├── test_risk_models.py
    ├── test_scenarios.py
    ├── test_attribution_euler.py
    ├── attribution/
    │   ├── test_brinson.py
    │   ├── test_engine.py
    │   ├── test_euler.py
    │   └── test_integration.py
    ├── backtest/
    │   ├── test_attribution_reporting.py
    │   ├── test_attribution_example.py
    │   └── test_brinson_utils_min.py
    ├── quick/                  # Fast smoke/property tests
    │   ├── test_brinson_coercer.py
    │   ├── test_engine_costs.py
    │   ├── test_engine_smoke.py
    │   ├── test_export_cli.py
    │   ├── test_kpis_smoke.py
    │   ├── test_metrics_smoke.py
    │   ├── test_plot_brinson_contract.py
    │   ├── test_plot_turnover.py        
    │   ├── test_reporting.py
    │   ├── test_reporting_build_smoke.py 
    │   ├── test_reporting_html.py
    │   ├── test_scenarios_bootstrap_and_slice.py
    │   ├── test_turnover_reconstruction.py 
    │   └── test_run_engine_smoke.py     
    └── unit/                   # Fine-grained unit tests
        ├── test_attribution_helpers.py
        ├── test_brinson_utils_edge_cases.py
        ├── test_brinson_utils_glue.py
        ├── test_brinson_utils_more.py
        ├── test_engine_expand.py
        ├── test_factor_decomposition.py  
        ├── test_kpis_edge_cases.py       
        ├── test_reporting_build_minimal.py
        ├── test_reporting_build_report_with_groups.py
        ├── test_reporting_context.py
        ├── test_reporting_render_html.py
        ├── test_reporting_render_html_brinson.py
        ├── test_reporting_render_html_full.py
        ├── test_reporting_render_html_metrics_only.py
        ├── test_reporting_render_html_minimal_ctx.py
        ├── test_reporting_render_html_no_figs.py
        ├── test_utils_ensure_psd_basic.py
        └── test_viz_attribution_plots.py  

---


## ⚙️ Tech Stack

| Component | Technology |
|----------|------------|
| **Frontend** | Streamlit + Plotly |
| **Numerical Backend** | Polars, NumPy, Pandas |
| **Optimization** | Custom PGD, HRP, TE optimizers |
| **Attribution** | Brinson, Euler |
| **Reporting** | Plotly + Jinja2 (HTML/PDF) |
| **Testing** | Pytest, Mypy, Ruff, Black, Pre-commit |
| **Environment** | Python ≥ **3.10** |


---


## 📊 Example Highlights

- **Backtest a portfolio with one line**
- **Compare scenarios (ΔCAGR, Sharpe, MaxDD)**
- **Tornado sensitivity analysis**
- **Historical replay (e.g., March 2020)**
- **Brinson/Euler attribution integrated into reporting**


---


## 🧮 Attribution Engine — Brinson & Euler (Overview)

- **Brinson Attribution** implemented in portfolio.attribution.brinson and *portfolio.backtest.brinson_utils*:
   • Normalize timeseries (wide / long / global-only) into a standard long format.

   • Compute allocation, selection, interaction and total effects per group and over the full period.

   • Build reporting-friendly structures with build_brinson_attribution_report, returning:
      • A long timeseries,
      • Aggregation by group,
      • And global totals

   • Euler Risk Contributions: Implemented in *portfolio.attribution.euler*:

   • Implemented in *portfolio.attribution.euler*:
      • Works on wide return/weight DataFrames (Polars),
	   •	Produces per-asset per-date contributions and portfolio-level aggregates,
	   •	Used both in unit tests and in higher-level backtest/reporting helpers.

These pieces are integrated into the backtest reporting module so that attribution becomes a first-class citizen in the final HTML/PDF reports.


---


## 🧪 Minimal Code Examples

1. Euler Risk Contributions
   ´´´python
   import numpy as np
   import pandas as pd
   from portfolio.attribution.euler import euler_risk_contributions

   # Portfolio weights
   w = pd.Series([0.6, 0.4], index=["A", "B"])

   # Simple covariance matrix
   cov = pd.DataFrame(
      [[0.04, 0.01],
      [0.01, 0.09]],
      index=["A", "B"],
      columns=["A", "B"],
   )

   rc = euler_risk_contributions(w, cov)

   print(rc)
   print("Sum of risk contributions:", rc.sum())
   ´´´

2. Brinson Attribution Reporting (Polars)
   ´´´python
   import polars as pl
   from portfolio.backtest.attribution_reporting import build_brinson_attribution_report

   # Example wide-style Brinson timeseries (2 dates × 2 groups)
   df = pl.DataFrame(
      {
         "date": ["2020-01-01", "2020-01-02"],
         "alloc_0": [0.1, 0.2],
         "alloc_1": [0.3, 0.4],
         "select_0": [0.5, 0.6],
         "select_1": [0.7, 0.8],
         "interact_0": [0.0, 0.0],
         "interact_1": [0.1, 0.2],
         "total_0": [0.6, 0.8],
         "total_1": [0.9, 1.0],
      }
   )

   report = build_brinson_attribution_report(df, how="sum")

   print(report["timeseries"])
   print(report["by_group"])
   print(report["total"])
   ´´´


---


## 🧠 Mathematical & Research Basis

GammaEdge builds upon:

- **Markowitz (1952)** — Mean-Variance  
- **Ledoit & Wolf (2003)** — Covariance shrinkage  
- **Lopez de Prado (2016)** — HRP  
- **Rockafellar & Uryasev (2002)** — CVaR optimization  
- **Brinson (1985)** — Performance attribution  
- **Euler decomposition** of portfolio volatility  

All models are implemented from first principles with numerical stability in mind.


---


## 🖥️ How to Run Locally

1. **Clone the repository and enter the project directory**
```bash
   git clone https://github.com/<your-username>/GammaEdge.git  
   cd GammaEdge  
```
2. **(Optional) Create a virtual environment**
```bash
   python -m venv .venv  
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate  
```
3. **Install dependencies**
```bash
   pip install -r requirements.txt  
```
4. **Run the Streamlit app**
```bash
   streamlit run app/app.py  
```
You can then navigate across the pages using the left sidebar (Data → Risk Model → Backtest → Scenarios → Attribution → Risk Analysis).


---

## 📘 Future Development

- ML-based factor forecasting (LSTM / TFT)  
- Intraday stress-testing  
- Multi-threaded optimizers  
- Real-time market ingestion dashboard  
- Integration with my [**Option Pricing Model**](https://github.com/Claudoi/OptionPricingModel) project to extend GammaEdge’s analytics into derivative pricing and volatility calibration.


---


## 🧾 License

This project is released under the **MIT License**.  
Feel free to use, adapt, or extend it for research, academic, or professional purposes — just include proper credit.


---
