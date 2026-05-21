# =============================================================================
# GammaEdge — Makefile
# Always uses the in-project venv (.venv/bin/*) to avoid Python version drift.
# =============================================================================

VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
POETRY   := poetry
STREAMLIT := $(VENV)/bin/streamlit
UVICORN  := $(VENV)/bin/uvicorn
PYTEST   := $(VENV)/bin/pytest
RUFF     := $(VENV)/bin/ruff
BLACK    := $(VENV)/bin/black
MYPY     := $(VENV)/bin/mypy

.PHONY: help install app api test test-fast lint format typecheck check \
        benchmark clean precommit precommit-install ci

# ─── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target>\n\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)

# ─── Setup ───────────────────────────────────────────────────────────────────

install: ## Install dependencies via Poetry into .venv
	$(POETRY) install

# ─── Run applications ───────────────────────────────────────────────────────

app: ## Launch the Streamlit app (Python 3.11 venv)
	$(STREAMLIT) run app/Home.py

api: ## Launch the FastAPI server with auto-reload
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

# ─── Tests ───────────────────────────────────────────────────────────────────

test: ## Run the full test suite
	$(PYTEST) tests/

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/ -p no:cov -o addopts="" --tb=short

# ─── Code quality ───────────────────────────────────────────────────────────

lint: ## Run ruff linter (auto-fixes where possible)
	$(RUFF) check --fix portfolio/ api/ app/ tests/

format: ## Run black formatter
	$(BLACK) portfolio/ api/ app/ tests/

typecheck: ## Run mypy on portfolio/
	$(MYPY) portfolio/

check: lint format typecheck test-fast ## Run all quality checks (lint + format + typecheck + tests)

# ─── Pre-commit ──────────────────────────────────────────────────────────────

precommit-install: ## Install pre-commit hooks into .git/hooks
	$(VENV)/bin/pre-commit install

precommit: ## Run pre-commit hooks against all files
	$(VENV)/bin/pre-commit run --all-files

# ─── Performance ────────────────────────────────────────────────────────────

benchmark: ## Run the EWMA covariance benchmark
	$(PY) scripts/benchmark_ewma.py

# ─── CI mirror (what GitHub Actions runs) ────────────────────────────────────

ci: ## Mirror the GitHub Actions CI workflow locally
	$(RUFF) check portfolio/ api/ app/ tests/
	$(BLACK) --check portfolio/ api/ app/ tests/
	$(MYPY) portfolio/
	$(PYTEST) tests/ --tb=short

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean: ## Remove caches (pytest, mypy, ruff, coverage, pycache)
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml 2>/dev/null || true
	@echo "✅ Caches cleaned"
