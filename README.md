# GammaEdge

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://gammaedge.streamlit.app)
![Polars](https://img.shields.io/badge/Polars-Fast-blue?logo=polars&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-65%25%2B-success?logo=codecov&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?logo=Open%20Source%20Initiative&logoColor=white)

> Python platform for institutional portfolio optimization, risk analytics, and backtesting.

Portfolio optimization and backtesting framework implementing modern quantitative methods. Includes covariance shrinkage, Random Matrix Theory filtering, multiple optimization engines, and vectorized backtesting with realistic transaction costs.

The implementation prioritizes numerical stability and addresses common issues with sample covariance matrices in limited time series. We use a multi-step fallback approach for ill-conditioned matrices and apply spectral filtering to separate signal from noise.

GammaEdge is developed as a final-degree project (TFG) for educational and research use at Universidad de Las Palmas de Gran Canaria.

---

## Features

### Risk Models
- Ledoit-Wolf and Oracle Approximating Shrinkage (OAS) for covariance estimation
- Random Matrix Theory (Marcenko-Pastur) eigenvalue denoising
- Black-Litterman framework with Idzorek's method for view incorporation

### Optimization
Seven optimizer implementations:
- **Mean-Variance**: Classic Markowitz with projected gradient descent
- **CVaR**: Conditional Value-at-Risk minimization via linear programming
- **Hierarchical Risk Parity**: Graph-based diversification without return forecasts
- **Risk Parity**: Equal risk contribution across assets
- **Tracking Error**: Minimize deviation from benchmark
- **Robust**: Uncertainty set-based optimization
- **Black-Litterman**: Bayesian combination of equilibrium and views

Constraint support: box limits, cardinality, sector exposure, turnover caps

### Backtesting
Vectorized backtesting engine with:
- Portfolio drift reconstruction between rebalances
- Transaction cost modeling (linear and non-linear)
- Almgren-Chriss square-root market impact
- Multiple rebalancing strategies (calendar-based, threshold-based)

### ML and Regime Detection
- Hidden Markov Models for regime identification (2 to 4 regimes)
- XGBoost predictors with SHAP interpretability
- Fama-French 3 and 5-factor model integration

### Analytics
- Brinson-Fachler attribution (Allocation, Selection, Interaction)
- Euler risk contributions for marginal risk analysis
- Historical scenario analysis (2008 crisis, Covid-19)
- Standard metrics: Sharpe, Sortino, Calmar, VaR, CVaR, etc.

---

## Quick start

### Prerequisites

- Python 3.11 (the project pins `>=3.11,<3.12` for CI reproducibility)
- Poetry 1.8+ (Poetry 2.x is also supported)

### Install

```bash
git clone https://github.com/Claudoi/GammaEdge.git
cd GammaEdge
poetry install
```

### Run the Streamlit app

```bash
poetry run streamlit run app/Home.py
```

The app launches with eight pages:

1. **Data** — ingest, clean and explore prices and returns
2. **Risk Model** — covariance estimation and RMT filtering
3. **Optimizer** — run the seven optimization engines
4. **Backtest** — vectorized simulation with transaction costs
5. **Attribution** — Brinson-Fachler decomposition
6. **Reporting** — PDF / Excel report generation
7. **Scenarios** — historical and synthetic stress tests
8. **Regime Detection** — HMM regime identification

### Run the REST API

```bash
poetry run uvicorn api.main:app --reload
```

Interactive documentation is available at `http://localhost:8000/docs` (Swagger UI)
and `http://localhost:8000/redoc` (ReDoc).

---

## Architecture

GammaEdge follows a three-layer architecture that separates pure quantitative
logic from presentation and integration concerns:

| Layer        | Path         | Purpose                                                            |
|--------------|--------------|--------------------------------------------------------------------|
| Core library | `portfolio/` | Pure quantitative logic (numpy / polars / scipy). Strict `mypy`.   |
| Web app      | `app/`       | Streamlit interface with eight pages.                              |
| REST API     | `api/`       | FastAPI endpoints + Pydantic v2 schemas for programmatic access.   |

Data flow: Polars long form -> hash-keyed cache -> numpy matrices -> optimizers
-> backtest -> attribution -> Streamlit / API.

```mermaid
graph TD
    A[Data Ingestion] --> B[Cleaning and Winsorization]
    B --> C[Risk Model]
    C --> D[Optimization Engine]
    D --> E[Vectorized Backtest]
    E --> F[Attribution Analysis]
    E --> G[Report Generation]

    style A fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style B fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style C fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style D fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style E fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style F fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style G fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
```

**Core libraries**: NumPy, SciPy, Polars, scikit-learn, statsmodels, hmmlearn, XGBoost
**Interface**: Streamlit application with 8 pages, FastAPI backend

---

## Usage

### Python example

```python
from portfolio.optim.cvar import solve_cvar_lp
from portfolio.features.risk_models import estimate_covariance, clean_covariance_rmt

# Covariance with shrinkage and RMT filtering
Sigma = estimate_covariance(returns, method="lw")
Sigma_clean = clean_covariance_rmt(Sigma, T=252, N=50)

# CVaR optimization (95% confidence)
weights = solve_cvar_lp(
    returns_matrix=returns,
    alpha=0.95,
    w_min=0.0,
    w_max=0.10,
)
```

### REST API

GammaEdge exposes a REST interface built on FastAPI alongside the Streamlit UI.

| Method | Path                  | Description                                                                  |
|--------|-----------------------|------------------------------------------------------------------------------|
| GET    | `/health`             | Liveness probe.                                                              |
| POST   | `/api/v1/optimize`    | Portfolio optimization (`mean_variance`, `min_variance`, `risk_parity`, `hrp`). |

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
        "returns": [[0.01, 0.02, -0.01], [0.005, 0.015, 0.008]],
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "method": "mean_variance",
        "rf": 0.04
      }'
```

The response includes per-ticker weights, annualized metrics (expected return,
volatility, Sharpe ratio) and a timestamp.

Input validation is enforced with Pydantic v2 models (`api/schemas/`), which
reject non-finite values, inconsistent dimensions, and duplicate tickers before
the optimizer is invoked.

---

## Development

### Run tests

```bash
poetry run pytest
```

The suite enforces a minimum coverage threshold of 65% on `portfolio/` and uses
Hypothesis for property-based testing of numerical kernels.

### Lint and format

```bash
poetry run ruff check --fix portfolio/ api/ app/ tests/
poetry run black portfolio/ api/ app/ tests/
```

### Type check

```bash
poetry run mypy portfolio/ api/
```

`app/` is intentionally excluded from strict typing to keep the Streamlit
prototyping loop fast.

### Pre-commit hooks

The repository uses pre-commit hooks (ruff + black on commit, mypy on demand):

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

---

## Implementation Notes

**Numerical Stability**: Six-step fallback chain for singular matrices (spectral clipping, iterative jitter, ridge regularization, diagonal loading, pseudoinverse, identity fallback)

**RMT Filtering**: Eigenvalue separation based on Marcenko-Pastur distribution to filter noise from sample covariance

**Transaction Costs**: Square-root model with permanent and temporary impact components

**Testing**: 65%+ coverage on core portfolio modules, property-based testing with Hypothesis

---

## References

Based on methods from:
- Markowitz (1952): Portfolio Selection
- Ledoit & Wolf (2004): Honey, I Shrunk the Sample Covariance Matrix
- Lopez de Prado (2016): Building Diversified Portfolios that Outperform Out of Sample
- Rockafellar & Uryasev (2000): Optimization of Conditional Value-at-Risk
- Almgren & Chriss (2001): Optimal Execution of Portfolio Transactions

---

## Citation

If you use GammaEdge in academic work, please cite:

```bibtex
@misc{martel2026gammaedge,
  author       = {Martel, Claudio},
  title        = {GammaEdge: A Portfolio Optimization Platform},
  year         = {2026},
  howpublished = {Final Degree Project (TFG), Universidad de Las Palmas de Gran Canaria},
}
```

---

**Author**: Claudio Martel
**License**: MIT (see `pyproject.toml`; project distributed under the MIT terms)
