# GammaEdge

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=Open%20Source%20Initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/badge/Last_Commit-Recently-brightgreen?logo=git&logoColor=white)](https://github.com/Claudoi/GammaEdge/commits/main)
[![Streamlit](https://img.shields.io/badge/Live%20App:%20Open%20in%20Streamlit-App-red?logo=streamlit&logoColor=white)](https://gammaedge.streamlit.app/)

# 🧠 GammaEdge — Portfolio Analytics, Risk Modeling & Scenario Simulation Platform

**GammaEdge** is an end-to-end research and visualization framework for **portfolio construction, backtesting, attribution and scenario analysis**.  
It combines academic-grade financial modeling with a clean, interactive **Streamlit interface**, allowing users to explore allocation methods, stress tests, and risk metrics in a single coherent environment.

This project started as a way to unify my quantitative finance tools — from covariance modeling to risk attribution — into a single, production-style platform that’s both **mathematically rigorous** and **visually intuitive**.

---

## 🚀 Key Features

- **Backtesting Engine**  
  Modular and allocator-agnostic, with full turnover accounting and transaction cost simulation.

- **Portfolio Optimization** — multiple allocators:
  - Equal Weight  
  - Risk Parity  
  - Hierarchical Risk Parity (HRP)  
  - Minimum Variance (L2-PGD)  
  - Tracking-Error Minimization vs Benchmark  
  - Robust & constrained optimizers (cardinality, turnover, etc.)

- **Scenario Analysis** — stress-test portfolios under:
  - Mean and covariance shocks  
  - One-day crashes  
  - Historical replay windows  
  - Beta-correlated market moves  
  - Tornado (one-at-a-time) sensitivity analysis  

- **Risk Metrics**  
  Full suite of diagnostics (CAGR, Sharpe, Max Drawdown, Turnover, hit ratio, etc.) plus portfolio-level VaR/ES modules.

- **Portfolio Attribution (New)**  
  A dedicated attribution engine that supports:
  - **Brinson-style attribution** (allocation / selection / interaction / total), with helpers to normalize timeseries and build reporting-ready tables.
  - **Euler risk contributions** from covariance matrices, giving asset-level risk decomposition that sums to total portfolio volatility.
  - Tight integration with the backtest engine and reporting layer.

- **Visualization Layer** — built on **Plotly**:
  - Equity & drawdown charts  
  - Correlation heatmaps and dendrograms  
  - Weight evolution and deltas  
  - Tornado sensitivity plots  
  - Attribution charts & top contributors  
  - Scenario-to-baseline comparisons  

- **Streamlit UI**  
  A professional multi-page design with clear sidebars, expander sections, and dynamic Plotly interactivity.

---

## 🧩 Folder Structure

```bash
GammaEdge/
│
├── README.md
├── pyproject.toml              # Build & dependency configuration (Poetry)
├── poetry.lock
├── requirements.txt            # Optional pip-based installation
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml     # Ruff, Black, nbstripout, etc.
├── .coveragerc
├── PR_CHECKLIST.md
├── CHANGELOG.md
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
│   └── optimizers.md
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
├── portfolio/                  # Core analytical backend
│   ├── __init__.py
│   ├── attribution/            # Attribution engine (Brinson & Euler)
│   │   ├── __init__.py
│   │   ├── brinson.py          # Brinson-style attribution helpers
│   │   ├── euler.py            # Euler risk contributions
│   │   └── engine.py           # Portfolio-level contributions engine
│   ├── backtest/               # Backtest & reporting
│   │   ├── __init__.py
│   │   ├── engine.py           # Core backtest engine
│   │   ├── reporting.py        # Plotly-based reporting (HTML/PDF)
│   │   ├── attribution.py      # Return/weight alignment & contributions
│   │   ├── attribution_reporting.py  # Brinson reporting helpers
│   │   ├── brinson_utils.py    # Low-level attribution utilities
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
    ├── test_kpis_property.py
    ├── test_mean_variance.py
    ├── test_optimizer_fallbacks.py
    ├── test_risk_models.py
    ├── test_scenarios.py
    ├── test_attribution_euler.py
    ├── test_attribution_integration.py
    ├── test_attribution_engine.py
    ├── test_brinson.py
    ├── backtest/
    │   ├── test_attribution_reporting.py
    │   └── test_brinson_utils_min.py
    ├── attribution/
    │   ├── test_brinson.py
    │   ├── test_engine.py
    │   └── test_euler.py
    ├── quick/                  # Fast smoke/property tests
    │   ├── test_engine_costs.py
    │   ├── test_engine_smoke.py
    │   ├── test_export_cli.py
    │   ├── test_kpis_smoke.py
    │   ├── test_metrics_smoke.py
    │   ├── test_plot_brinson_contract.py
    │   └── test_reporting.py
    └── unit/                   # Fine-grained unit tests
        ├── test_attribution_helpers.py
        ├── test_brinson_utils_edge_cases.py
        ├── test_brinson_utils_glue.py
        ├── test_brinson_utils_more.py
        ├── test_engine_expand.py
        ├── test_reporting_build_minimal.py
        ├── test_reporting_render_html.py
        ├── test_reporting_render_html_brinson.py
        ├── test_reporting_render_html_full.py
        ├── test_reporting_render_html_metrics_only.py
        ├── test_reporting_render_html_minimal_ctx.py
        ├── test_reporting_render_html_no_figs.py
        └── test_utils_ensure_psd_basic.py

´´´

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend / UI | Streamlit + Plotly |
| Core Analytics | NumPy, Polars, Pandas |
| Optimization | Custom Python (HRP, L2-PGD, Risk Parity, TE) |
| Attribution | Brinson-style & Euler risk contributions) |
| Risk & Metrics | Custom metrics engine + financial math |
| Testing & Dev | Pytest, Mypy, Ruff, Black, Pre-commit |
| Environment | Python ≥ 3.9 |

---

## 📊 Example Highlights

- **Baseline Backtest**: Compute portfolio weights over time and analyze turnover.  
- **Scenario Comparison**: Apply mean/covariance shocks or crisis replays, and compare key metrics (ΔCAGR, Sharpe, MaxDD).  
- **Tornado Sensitivity**: Evaluate which assets have the strongest up/down impact on portfolio CAGR.  
- **Historical Slices**: Replay specific market periods (e.g., March 2020) with consistent allocator logic.  
- **Rebalance Diagnostics**: Inspect allocator calls, unique weight changes, and turnover per step.
- **Portfolio Attribution**: 	
	•	Compute Brinson-style allocation/selection/interaction effects by group.
	•	Decompose portfolio risk into Euler contributions per asset.
	•	Plug attribution results into the backtest reporting layer (tables + Plotly figures).

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

GammaEdge draws inspiration from:

- **Markowitz (1952)** — Mean-Variance Optimization  
- **Ledoit & Wolf (2003)** — Shrinkage Covariance Estimation  
- **Lopez de Prado (2016)** — Hierarchical Risk Parity  
- **Cont (2001)** — Financial Turbulence and Stress Testing  
- **Rockafellar & Uryasev (2002)** — Conditional Value-at-Risk Optimization  
- **Classic Brinson-style** attribution frameworks and Euler decompositions of risk

Every formula and method is implemented from first principles, with emphasis on numerical stability and interpretability.

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

Planned extensions include:

- Risk attribution by factor decomposition (Brinson-Fachler, Euler decomposition).  
- Advanced covariance modeling (DCC-GARCH, shrinkage estimators).  
- Monte Carlo VaR with dynamic factor structures.  
- GPU acceleration for optimization and scenario runs.  
- Interactive dashboard for real-time market data ingestion.  
- Integration with my [**Option Pricing Model**](https://github.com/Claudoi/OptionPricingModel) project to extend GammaEdge’s analytics into derivative pricing and volatility calibration.

---

## 🧾 License

This project is released under the **MIT License**.  
Feel free to use, adapt, or extend it for research, academic, or professional purposes — just include proper credit.

---
