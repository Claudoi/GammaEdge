# Changelog

## 2.2.0 — 2026-05-21

### Security
- **Rotated dataset signing key** (`keys/gammaedge_release_ed25519`).
  The previous Ed25519 private key was accidentally committed to git history
  in January 2026 (commit `ccf9032`) and pushed to the public repository.
  A new Ed25519 keypair has been generated; the previous key is now revoked
  for new dataset signatures.
  Public key (`keys/gammaedge_release_ed25519.pub`) and the `keys/allowed_signers`
  trust file have been updated. Datasets re-signed after this rotation must be
  verified against the new public key.
  Going forward, the project `.gitignore` blocks any private key under `keys/`
  (`keys/*_ed25519`, `keys/*_rsa`, `keys/*_ecdsa`, `*.pem`, `secrets/`, `.env`).

### Added
- Comprehensive `.gitignore` covering Python caches, virtualenvs, IDE files,
  OS artifacts, LaTeX compile outputs, secrets, logs, generated data, and
  internal engineering docs.
- `poetry.lock` tracked for reproducible installs.
- `CLAUDE.md` and `.claude/settings.json` for Claude Code workflows.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`):
  ruff, black, mypy, pytest under Poetry with venv caching.
- REST API: `POST /api/v1/optimize` (mean-variance, min-variance, risk parity,
  HRP) and `GET /health`. Pydantic v2 schemas with input validation;
  integration tests via FastAPI `TestClient`.
- Test coverage for `portfolio/attribution/` (Euler + Brinson-Fachler) and
  `portfolio/backtest/reporting/` (previously 0%).
- Expanded README with architecture, quick start, usage examples, and
  citation block.

### Changed
- Unified `ensure_psd()` into a single canonical implementation
  (`portfolio/core/utils.py`); removed duplicates in `optim/mean_variance.py`
  and `features/risk_models.py`.
- Vectorized EWMA covariance (`portfolio/features/risk_models.py`):
  ~70x speedup vs. the previous Python loop on T=500, N=20.
- Standardized session key naming across Streamlit pages: `returns_wide`
  is the canonical key; `df_ret_wide` fallback removed.
- All Streamlit pages now apply the project Plotly dark theme via
  `app/viz/plotly_theme.py::apply_gammaedge_theme`.
- Migrated deprecated pandas frequency aliases (`M`/`Q`/`Y` → `ME`/`QE`/`YE`)
  in `portfolio/features/returns.py`.
- `data_loader.get_prices_long/wide` return Polars consistently
  (removed pandas round-trip in the cache layer).
- Lazy import of `pandas_datareader` in `factor_models.py` to unblock test
  collection.

### Fixed
- Backtest engine returned NaN equity curves when any ticker had leading NaN
  rows; engine now truncates to the first common valid date and logs a warning.
- CVaR LP solver failure now returns an equal-weight fallback instead of
  raising `RuntimeError`.
- `yfinance` silently dropped missing tickers; loader now raises `ValueError`
  listing the absent symbols.
- Ledoit-Wolf and OAS covariance estimators are now explicitly projected to
  the PSD cone after fit to guard against numerical noise.
- HMM `RegimeDetector.fit()` validates a minimum number of observations
  (`max(100, n_regimes * 30)`) and raises a clear error otherwise.
- `expected_returns()` raises `ValueError` listing affected assets when the
  result would contain NaN or Inf.
- Single-asset frontier (`frontier_closed_form` with `n <= 1`) now raises
  `ValueError` with guidance.
- HRP `_split_allocation` index re-mapping bug for `N >= 3` (recursive
  submatrix slicing was passing parent-level indices).
- `_wide_to_matrix` now writes to a writable numpy copy
  (the read-only pandas view caused `ValueError` on NaN assignment).
- Multiple silent `except: pass` blocks in Streamlit pages 02/03/07
  replaced with structured logging.
- Pre-existing `04_Backtest.py` crash when the engine returned an error
  payload — now handled with a user-facing warning.

### Removed
- Emojis from Streamlit page titles/headers/messages (project style policy).
- Legacy `_make_returns` helper in risk model tests (dead code).
- `keys/gammaedge_release_ed25519` (private key) from tracking.
- Generated artifacts from tracking: `reports/`, `results/`, `data_lake/raw/`,
  `datasets/` (re-fetchable / regenerable).

### Documentation
- LaTeX report (`LaTeX.txt`):
  - Corrected the optimizer list (removed the unimplemented "Genetic" claim).
  - Rewrote the `ensure_psd` description to match the actual spectral-clip
    implementation (the prior 7-stage fallback description was aspirational).
  - Added a footnote noting `RF_ANNUAL_DEFAULT = 0.04` in the Sharpe section.
  - Added `\begin{remark}` blocks on numerical stability, data completeness
    in the backtest engine, and the REST API as implemented (not future work).

### Future Work (tracked, not blocking)
Items identified during the v2.2.0 audit that are deferred to a later sprint:

- **Module split — `portfolio/viz/plot_utils.py`** (2492 lines, 60 functions)
  is planned to be split into four focused modules: `correlation_plots`,
  `portfolio_plots`, `scenario_plots`, `backtest_plots`.
- **Module split — `portfolio/trading/`** (3000+ LOC, v1 and v2 allocation
  variants plus ML predictors) is planned to be reorganized into
  `portfolio/strategies/` and `portfolio/ml/`; legacy v1 will be archived.
- **Accessibility**: custom HTML components (`metric_grid`, `data_hero_card`)
  lack ARIA labels and Plotly charts are not exposed to screen readers.
  Keyboard navigation hints are planned.
- **Responsive design**: 4-column layouts in pages `01_Data` and `02_Risk`
  do not degrade well below ~800 px. A media query in
  `app/viz/styles.get_global_styles()` is planned.
- **REST API**: the `/api/v1/backtest` endpoint is still a stub returning
  `{"backtest": True}`; async handlers, rate limiting, and cache headers
  are planned for v2.3.0.
- **Cache layer**: the file-based Parquet cache has a small race window
  (lock acquired after the existence check) and lacks TTL/LRU eviction
  and on-load schema validation. A migration path to Redis is documented
  as a known limitation.
- **Persistence**: there is no relational database; multi-user access,
  audit trails, and resumable jobs are out of scope for the single-user
  TFG release.
- **Rolling-window perf**: `portfolio/viz/quant_charts.py` (rolling Sharpe,
  lines 299-320) still uses an explicit Python loop. A `pandas.rolling()`
  rewrite is a low-impact micro-optimization.
- **Streamlit caching**: `app/pages/05_Attribution.py` does not wrap the
  Brinson decomposition in `@st.cache_data`. Only relevant if decomposition
  cost exceeds ~500 ms in practice.

## 2.0.0 — stable
- CI verde: ruff, black, mypy en portfolio y api, pytest con cobertura 75.22%
- Cobertura mínima fijada en 65%
- Limpieza de typing y hooks pre-commit
- Aislamiento de app/ fuera de mypy para estabilizar release
- pyproject consolidado para Poetry 1.8.x
