# GammaEdge

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=Open%20Source%20Initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/badge/Last_Commit-Recently-brightgreen?logo=git&logoColor=white)](https://github.com/Claudoi/GammaEdge/commits/main)
[![Streamlit](https://img.shields.io/badge/Live%20App:%20Open%20in%20Streamlit-App-red?logo=streamlit&logoColor=white)](https://gammaedge.streamlit.app/)

# 🧠 GammaEdge — Portfolio Analytics, Risk Modeling & Scenario Simulation Platform

**GammaEdge** is an end-to-end research and visualization framework for **portfolio construction, backtesting, and scenario analysis**.  
It combines academic-grade financial modeling with a clean, interactive **Streamlit interface**, allowing users to explore allocation methods, stress tests, and risk metrics in a single coherent environment.

This project started as a way to unify my quantitative finance tools — from covariance modeling to risk attribution — into a single, production-style platform that’s both **mathematically rigorous** and **visually intuitive**.

---

## 🚀 Key Features

- **Backtesting Engine** — Modular and allocator-agnostic, with full turnover accounting and transaction cost simulation.  
- **Portfolio Optimization** — Includes multiple allocators:
  - Equal Weight  
  - Risk Parity  
  - Hierarchical Risk Parity (HRP)  
  - Minimum Variance (L2-PGD)  
  - Tracking-Error Minimization vs Benchmark  
- **Scenario Analysis** — Stress-test portfolios under:
  - Mean and covariance shocks  
  - One-day crashes  
  - Historical replay windows  
  - Beta-correlated market moves  
  - Tornado (one-at-a-time) sensitivity analysis  
- **Risk Metrics** — Full suite of diagnostics (CAGR, Sharpe, Max Drawdown, Turnover, etc.) plus portfolio-level VaR/ES modules.  
- **Visualization Layer** — Built on **Plotly** with adaptive dark/light themes:
  - Equity & drawdown charts  
  - Correlation heatmaps and dendrograms  
  - Weight evolution and deltas  
  - Tornado sensitivity plots  
  - Scenario-to-baseline comparisons  
- **Streamlit UI** — A professional multi-page design with clear sidebars, expander sections, and dynamic Plotly interactivity.




## 🧩 Folder Structure

GammaEdge/
│
├── .dockerignore
├── .gitignore
├── README.md
├── pyproject.toml              # Build & dependency configuration
├── requirements.txt            # Python dependencies
│
├── .github/                    # CI/CD workflows
│   └── workflows/
│       └── ci.yml
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
│       └── 06_Scenarios.py
│
├── data/                       # Sample datasets
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
│   ├── viz/
│   │   └── plot_utils.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── reporting.py
│   │   ├── attribution.py
│   │   └── metrics.py
│   ├── features/
│   │   ├── factors.py
│   │   ├── returns.py
│   │   ├── risk_models.py
│   │   └── scenarios.py
│   ├── io/
│   │   ├── cache.py
│   │   └── data_loader.py
│   ├── core/
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── opt_helpers.py
│   │   └── utils.py
│   └── optim/
│       ├── hrp.py
│       ├── mean_variance.py
│       ├── black_litterman.py
│       ├── cvar.py
│       ├── risk_parity.py
│       └── te.py
│
└── tests/                      # Unit tests
    ├── test_backtest.py
    ├── test_core_utils.py
    ├── test_mean_variance.py
    └── test_risk_models.py



## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend / UI | Streamlit + Plotly |
| Core Analytics | NumPy, Polars, Pandas |
| Optimization | Custom Python (HRP, L2-PGD, Risk Parity) |
| Risk & Metrics | Custom metrics engine + financial math |
| Testing & Dev | Pytest, Pre-commit hooks |
| Environment | Python ≥ 3.9 |

---

## 📊 Example Highlights

- **Baseline Backtest**: Compute portfolio weights over time and analyze turnover.  
- **Scenario Comparison**: Apply mean/covariance shocks or crisis replays, and compare key metrics (ΔCAGR, Sharpe, MaxDD).  
- **Tornado Sensitivity**: Evaluate which assets have the strongest up/down impact on portfolio CAGR.  
- **Historical Slices**: Replay specific market periods (e.g., March 2020) with consistent allocator logic.  
- **Rebalance Diagnostics**: Inspect allocator calls, unique weight changes, and turnover per step.

---

## 💡 Motivation

I built **GammaEdge** to bring academic portfolio theory closer to how real-world allocators think.  
Instead of treating optimization, risk, and backtesting as separate scripts, GammaEdge integrates them into a single modular system — allowing me to **experiment, visualize, and debug allocation logic interactively**.

It also serves as a bridge between **quantitative research** and **risk management**: every page reflects a different analytical dimension of a professional multi-asset portfolio workflow.

---

## 🧠 Mathematical & Research Basis

GammaEdge draws inspiration from:

- **Markowitz (1952)** — Mean-Variance Optimization  
- **Ledoit & Wolf (2003)** — Shrinkage Covariance Estimation  
- **Lopez de Prado (2016)** — Hierarchical Risk Parity  
- **Cont (2001)** — Financial Turbulence and Stress Testing  
- **Rockafellar & Uryasev (2002)** — Conditional Value-at-Risk Optimization  

Every formula and method is implemented from first principles, with emphasis on numerical stability and interpretability.

---

## 🖥️ How to Run Locally

1. **Clone the repository and enter the project directory**

   git clone https://github.com/<your-username>/GammaEdge.git  
   cd GammaEdge  

2. **(Optional) Create a virtual environment**

   python -m venv .venv  
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate  

3. **Install dependencies**

   pip install -r requirements.txt  

4. **Run the Streamlit app**

   streamlit run app/app.py  

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

## 👋 About Me

I’m currently pursuing a dual degree in **Accounting & Finance** and **Computational Mathematics**, with a focus on **Quantitative Finance and Risk Modeling**.  
GammaEdge reflects my passion for building tools that combine theory, data, and clean design — a personal sandbox for experimenting with quantitative ideas before applying them in real financial systems.

If you find this project interesting, feel free to connect or collaborate on GitHub or LinkedIn.

---