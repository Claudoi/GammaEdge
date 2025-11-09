# 📈 Backtesting & Attribution in GammaEdge

This page explains how to use GammaEdge's **backtesting engine** together with the new **Brinson-style performance attribution** and **Euler risk contributions**.

It is meant as a practical guide for:
- Running a backtest on a universe of assets.
- Computing daily and total contributions by asset / group.
- Using the Brinson attribution helpers for reporting.
- Understanding how the attribution integrates into the reporting layer.

---

## 🧱 Building Blocks

GammaEdge’s attribution stack is composed of:

- `portfolio.attribution.engine`
  - Core portfolio-level contributions engine.
  - Works on **wide** Polars dataframes (`date` + one column per asset).
- `portfolio.attribution.brinson`
  - Helpers to work with **Brinson-style attribution timeseries** (alloc / select / interact / total).
  - Normalizes long/wide/global input formats.
- `portfolio.attribution.euler`
  - Euler risk contributions for variance-based risk, using covariances.
- `portfolio.backtest.attribution`
  - Glue between backtest engine and attribution (per-asset and per-group contributions).
- `portfolio.backtest.attribution_reporting`
  - High-level helper to build a **Brinson attribution report** (timeseries / by-group / total).
- `portfolio.backtest.reporting`
  - General backtest reporting: equity, drawdown, weights heatmap, and attribution sections.

If you just want to use the **high-level API**, you will mainly interact with:

- `portfolio.backtest.engine`
- `portfolio.backtest.attribution`
- `portfolio.backtest.attribution_reporting`
- `portfolio.backtest.reporting`

---

## 🔁 Typical Workflow Overview

1. Prepare **price / returns data** in wide format (one column per asset).
2. Choose or implement a **strategy / allocator** (e.g. risk parity, Markowitz, etc.).
3. Run the **backtest engine** to get:
   - Dates
   - Weights over time
   - Equity curve
4. Compute **attribution**:
   - By asset (daily + total contributions).
   - Optionally by group (sector, country, style, etc.).
5. Optionally compute **Brinson-style attribution** over groups.
6. Use the **reporting helpers** to build tables + Plotly figures and export:
   - HTML reports
   - PDF reports

---

## 📊 Data Requirements

### Returns (wide format, Polars)

The backtest and attribution engines expect a Polars DataFrame like:

```python
import polars as pl

df_ret_wide = pl.DataFrame(
    {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "AAPL": [0.01, -0.005, 0.002],
        "MSFT": [0.008, -0.003, 0.004],
        "GOOG": [0.012, -0.006, 0.001],
    }
)
