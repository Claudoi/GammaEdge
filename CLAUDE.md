# GammaEdge — Portfolio Optimization, Risk Analytics & Backtesting Platform

> Python platform implementing modern quant methods: covariance shrinkage, RMT filtering,
> 7 optimizer engines, vectorized backtesting, Brinson-Fachler attribution, HMM regime detection.
> Institutional quality. Bloomberg-level density.

## User Context

- **Role**: Investment Analyst / Quant Developer
- **Background**: Accounting & Finance + Computational Mathematics
- **Language**: Communicate in Spanish
- **Standard**: Bloomberg-level density. Professional dark-theme aesthetic.
- **Solo developer** — self-review standards must be high

## Commands

```bash
# App
streamlit run app/Home.py

# API
uvicorn api.main:app --reload

# Install
poetry install

# Test
pytest                          # 65%+ coverage on portfolio/

# Lint + format (run before every commit)
ruff check --fix . && black .

# Type check (CI only — slow, not in pre-commit)
mypy portfolio api
```

## Architecture

```
GammaEdge/
├── app/
│   ├── Home.py                 # Streamlit entry + Plotly compat monkey-patch
│   ├── design_system.py        # COLORS, get_global_styles — single source for UI
│   ├── utils.py
│   ├── viz/plotly_theme.py     # Plotly dark theme constants
│   └── pages/
│       ├── 01_Data.py          # Data ingestion + market data browser
│       ├── 02_RiskModel.py     # Covariance models (LW, OAS, RMT, EWMA)
│       ├── 03_Optimizer.py     # 7 optimizer implementations
│       ├── 04_Backtest.py      # Vectorized backtesting + transaction costs
│       ├── 05_Attribution.py   # Brinson-Fachler + Euler risk contributions
│       ├── 06_Reporting.py     # PDF/Excel report generation
│       ├── 07_Scenarios.py     # Historical scenario analysis
│       └── 08_RegimeDetection.py  # HMM regime identification
├── portfolio/                  # Core library — strict mypy, fully typed
│   ├── attribution/            # Brinson-Fachler + Euler contributions
│   ├── backtest/               # Vectorized engine + sqrt market impact
│   ├── core/                   # Metrics, guards, validators, compat, utils
│   ├── features/               # quant_metrics, risk_models, regime_detection, factor_models
│   ├── io/                     # Data loading, hash-based cache, normalization, ingestion
│   ├── optim/                  # MV, CVaR, HRP, RP, BL, TE, Robust — all with EW fallback
│   ├── trading/
│   └── viz/
├── api/                        # FastAPI — routes are stubs (not production-ready)
├── configs/                    # example_markowitz.yaml, example_blacklitterman.yaml
├── data_lake/raw/              # Raw market data
├── tests/                      # pytest + hypothesis, 65%+ coverage on portfolio/
└── pyproject.toml              # black (100), ruff (I,E,F,UP,SIM,B,C90), mypy, pytest
```

**Data flow:**

```
Polars long [date, ticker, price]
  → portfolio/io/    cache (hash-keyed, atomic writes)
  → numpy matrices   (mu, Sigma, w as float64 arrays)
  → portfolio/optim/ optimizers → equal-weight fallback on any failure
  → portfolio/backtest/ sqrt impact model
  → portfolio/attribution/
  → app/design_system.py → Streamlit pages
```

**Layer responsibilities:**

| Layer | Responsibility | Mypy |
|-------|---------------|------|
| `portfolio/` | Pure quant logic | Strict (all defs typed, no implicit Any) |
| `api/` | FastAPI endpoints | Lenient (early stage, route stubs) |
| `app/` | Streamlit UI only | Excluded by design |
| `tests/` | pytest + hypothesis | — |

## Conventions

- **Lazy imports on every Streamlit page** — never import heavy modules at top level (prevents startup crashes; see PR #71 cryptography fix)
- **Polars** for I/O and data transforms; **numpy** for all matrix math in `portfolio/`; **pandas** only for sklearn/statsmodels interop
- `app/design_system.py` is the only source of truth for colors and styles — never inline CSS
- `portfolio/core/compat.py` — use `@dataclass_compat` / `@dataclass_frozen_slots` instead of raw `@dataclass` (slots compatibility across Python versions)
- `portfolio/core/validators.py` — always validate results through `validate_annual_metrics()` before display (catches double-annualization; max return threshold 300%, max vol 200%)
- Every optimizer has an equal-weight fallback — match this pattern when adding new optimizers
- `logger = logging.getLogger(__name__)` in every module; never use `print()`
- Line length: 100 chars (black + ruff)
- Type hints on all `portfolio/` functions — Python 3.11+ syntax (`X | None`, not `Optional[X]`)

## Quantitative Rules

| Rule | Detail |
|------|--------|
| Covariance default | **OAS** (`portfolio/features/risk_models.py:273`); LW as fallback for insufficient data |
| Covariance requirement | Never use raw sample covariance — always shrinkage or RMT filtering |
| Portfolio weights | Long-only: `w ∈ [0, 1]`, enforce `∑w = 1.0 ± 1e-6` via `project_to_box_simplex()` |
| Return frequency | Annualize: `√252` daily · `√52` weekly · `√12` monthly (`TRADING_DAYS_PER_YEAR = 252`) |
| Sharpe | Excess returns over `RF_ANNUAL_DEFAULT = 0.04` (4% annual); document if using different rf |
| CVaR | Default `α = 0.95` (`portfolio/optim/cvar.py:13`); document deviations |
| Transaction costs | Sqrt impact model: `impact ∝ trade^1.5 / √vol` — NOT full Almgren-Chriss |
| HMM default | 3 states (`n_regimes=3`); 4-state supported since v1.1.0; validate with AIC/BIC before changing |
| Fama-French | Via `pandas_datareader`; handle missing data gracefully |

## Behavior Rules

1. **Plan First** — enter planning mode for any non-trivial task (>3 steps or architectural decisions)
2. **Subagents** — max 3 parallel; one task per agent; delegate research and exploration
3. **Self-Improvement** — after any correction, note the pattern in `docs/lessons.md`
4. **Verify Before Done** — never mark complete without proving it works. "Would a Staff Engineer approve this?"
5. **Fix Bugs Autonomously** — when given a bug report, fix it; don't ask for step-by-step guidance
6. For trivial tasks, use judgment — these rules bias toward caution over speed

## Core Principles

- **Simplicity**: Minimum necessary code. Touch only what's needed.
- **No Laziness**: Root causes, not temporary fixes. Senior developer standards.
- **Information Density**: Bloomberg, not Robinhood. Every pixel counts.
- **Dark Theme**: Professional aesthetic, monospace numbers, green/red coding.
- **Alpha Focus**: Only implement metrics that generate alpha. No decorative indicators.

## Git Conventions

- **Branch naming**: `feature/{description}`, `fix/{description}`
- **Commit format**: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`)
- **Merge strategy**: Squash-and-merge

## Critical Rules

- Always use lazy imports on Streamlit pages — never top-level heavy module imports
- Never write display or UI logic in `portfolio/` — belongs in `app/pages/`
- Never bypass `app/design_system.py` with inline styles
- Never add an optimizer without a corresponding test and equal-weight fallback
- Never use raw sample covariance (`method="sample"`) in production paths
- Never add `# type: ignore` in `portfolio/` without an explanatory comment
- Always run `ruff check --fix . && black .` before committing (mypy is CI-only)

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus. Use GitNexus MCP tools to understand code,
assess impact, and navigate safely before editing.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal.

## Always Do

- **Run impact analysis before editing any symbol.** `gitnexus_impact({target: "symbolName", direction: "upstream"})` — report blast radius to user.
- **Run `gitnexus_detect_changes()` before committing** — verify changes only affect expected symbols.
- **Warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows
2. `gitnexus_context({name: "<suspect function>"})` — see callers, callees, flow participation
3. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})`

## When Refactoring

- **Renaming**: `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first
- **Extracting**: `gitnexus_context()` + `gitnexus_impact()` before moving code
- After any refactor: `gitnexus_detect_changes({scope: "all"})` to verify scope

## Never Do

- NEVER edit a function/class without first running `gitnexus_impact`
- NEVER ignore HIGH or CRITICAL risk warnings
- NEVER rename symbols with find-and-replace — use `gitnexus_rename`

## Tools Quick Reference

| Tool | When to use |
|------|-------------|
| `gitnexus_query({query: "..."})` | Find code by concept |
| `gitnexus_context({name: "..."})` | 360° view of one symbol |
| `gitnexus_impact({target: "...", direction: "upstream"})` | Blast radius before editing |
| `gitnexus_detect_changes({scope: "staged"})` | Pre-commit scope check |
| `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` | Safe rename |

## Keeping the Index Fresh

After committing code changes, re-run:
```bash
npx gitnexus analyze
# If embeddings exist:
npx gitnexus analyze --embeddings
```
Check `.gitnexus/meta.json` → `stats.embeddings` to know if embeddings exist.

## Skill Files

| Task | Skill |
|------|-------|
| Understand architecture | `gitnexus-exploring` |
| Blast radius analysis | `gitnexus-impact-analysis` |
| Trace bugs | `gitnexus-debugging` |
| Rename / refactor | `gitnexus-refactoring` |
| CLI commands | `gitnexus-cli` |

<!-- gitnexus:end -->
