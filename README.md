# GammaEdge
### Institutional-Grade Quantitative Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-Fast-blue?logo=polars&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-65%25%2B-success?logo=codecov&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?logo=Open%20Source%20Initiative&logoColor=white)

**GammaEdge** is a production-ready computational framework designed to bridge the gap between **academic theory** and **institutional reality**. It unifies advanced portfolio optimization, robust risk modeling, and realistic backtesting into a single, mathematically rigorous ecosystem.

Unlike standard libraries that assume ideal market conditions, GammaEdge is engineered for **financial reality**: it handles ill-conditioned covariance matrices, non-normal return distributions, market impact costs, and structural regime changes natively.

---

## 🎯 The Philosophy: "Robustness is Alpha"

Quantitative finance is not about finding the "perfect" portfolio in-sample; it is about minimizing the **estimation error** that destroys out-of-sample performance.

GammaEdge addresses the core failures of naive quantitative implementations:
1.  **Numerical Stability**: Where standard optimizers fail on singular matrices, our `ensure_psd` engine employs a **6-step fallback chain** (including spectral clipping and iterative jitter) to guarantee convergence.
2.  **Signal vs. Noise**: We utilize **Random Matrix Theory (Marcenko-Pastur)** to surgically filter noise from covariance matrices, preventing the optimizer from allocating capital to spurious correlations.
3.  **Realistic Friction**: Our backtesting engine discards the "zero-cost" fantasy, implementing an institutional **Square-Root Impact Model** (Almgren-Chriss) and rigorous turnover accounting.

---

## 🚀 Key Capabilities

### 1. Advanced Risk Modeling
*   **Covariance Shrinkage**: Implementation of **Ledoit-Wolf** and **Oracle Approximating Shrinkage (OAS)** to stabilize estimators in high dimensions ($N \approx T$).
*   **Noise Filtering**: **RMT Denoising** ensures that the eigenspectrum of your risk model reflects true market structure, not sampling noise.
*   **Black-Litterman**: A Bayesian framework utilizing **Idzorek's method** to integrate subjective views with market equilibrium, allowing for calibrated confidence levels.

### 2. Convex Optimization Engine
*   **Beyond Mean-Variance**: Full support for **CVaR (Conditional Value-at-Risk)** optimization via linear programming for tail-risk management.
*   **Hierarchy-Based**: **Hierarchical Risk Parity (HRP)** (López de Prado) utilizes graph theory to build diversified portfolios without requiring unstable return forecasts.
*   **Risk Parity**: Construction of "All-Weather" style portfolios that equalise risk contributions across assets or factors.
*   **Constraints**: Solver-agnostic implementation of box constraints, cardinality limits, and sector exposure bounds via **Projected Gradient Descent (PGD)**.

### 3. Institutional Backtesting
*   **Vectorized Speed**: A high-performance engine capable of simulating decades of daily changes in milliseconds, powered by **Polars** and **NumPy**.
*   **Drift Reconstruction**: Accurate modelling of portfolio drift between rebalance periods to capture exact turnover requirements.
*   **Transaction Costs**: Configurable linear and non-linear cost models to simulate liquidity constraints and market impact.

### 4. TIER 1 Alpha Research (New)
*   **Regime Detection**: **Hidden Markov Models (HMM)** to identify latent market states (Volatile/Trending) and adapt strategies dynamically.
*   **ML Predictors**: **XGBoost** integration with **SHAP** explainability for non-linear return forecasting.
*   **Factor Models**: Native integration of **Fama-French 3 & 5 Factor** models for alpha decomposition.

### 5. Attribution & Analytics
*   **Brinson-Fachler**: Exact decomposition of active return into Allocation, Selection, and Interaction effects.
*   **Euler Risk Contributions**: Marginal contribution to portfolio volatility metrics, enabling precise risk budgeting.
*   **Scenario Analysis**: Stress-testing portfolios against historical crashes (2008, Covid-19) and synthetic shocks.

---

## 🏗 System Architecture

GammaEdge is built as a modular functional pipeline, emphasizing data immutability and type safety.

```mermaid
graph LR
    Data[Data Ingestion<br/>(Yahoo/S3)] --> Cleaning[Statistical Cleaning<br/>(Winsorization/Imputation)]
    Cleaning --> Risk[Risk Model<br/>(Shrinkage/RMT)]
    Risk --> Optim[Optimizer<br/>(CVaR/MVO/HRP)]
    Optim --> Backtest[Backtest Engine<br/>(Vectorized/Costs)]
    Backtest --> Attribution[Attribution<br/>(Brinson/Euler)]
    Backtest --> Reporting[Reporting<br/>(HTML/PDF)]
```

### Stack Highlights
*   **Core**: Python 3.11+, NumPy, SciPy
*   **Data**: Polars (High-performance DataFrames), yfinance
*   **ML/Stats**: scikit-learn, statsmodels, hmmlearn, XGBoost
*   **Interface**: Streamlit (Apple-style design system), Plotly, FastAPI

---

## ⚡ Quick Start

### Prerequisites
*   Python 3.11 or 3.12 (Recommended)
*   [Poetry](https://python-poetry.org/) (Recommended for dependency management)

### Installation

```bash
# Clone the repository
git clone https://github.com/Claudoi/GammaEdge.git
cd GammaEdge

# Install dependencies via Poetry
poetry install

# Launch the Quant Platform
poetry run streamlit run app/Home.py
```

### Example Usage: Optimal Portfolio Construction

```python
from portfolio.optim.cvar import solve_cvar_lp
from portfolio.features.risk_models import estimate_covariance, clean_covariance_rmt
import numpy as np

# 1. Robust Estimation
Sigma_noisy = estimate_covariance(returns, method="lw")
Sigma_clean = clean_covariance_rmt(Sigma_noisy, T=252, N=50)

# 2. CVaR Optimization (Minimizing 95% Tail Risk)
# We minimize Expected Shortfall (CVaR) instead of Variance for robust downside protection.
weights = solve_cvar_lp(
    returns_matrix=returns,
    alpha=0.95,
    w_min=0.0,
    w_max=0.10  # 10% Position Limit
)

print(f"Optimal Allocation: {weights}")
```


---

**Developed by Claudio Martel**
*Based on research by Markowitz, Ledoit-Wolf, López de Prado, and Rockafellar.*
