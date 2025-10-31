# Scenarios + Attribution: stable UI, tests ≥60% coverage

## Summary
- 07_Scenarios: plots sin show_plot, claves únicas para evitar colisiones en Streamlit.
- Métricas robustas con fallback y reconstrucción de turnover por drift entre rebalanceos.
- Comparativas Baseline vs Escenario: equity, drawdown, heatmaps de pesos y delta de CAGR.
- Suite verde local con cobertura ≥ 60 %.

## What’s included
- app/pages/07_Scenarios.py revisado: allocator factory, engine wrapper con recorder, métricas seguras, beta-shock, historical slice, tornado sensitivity y descargas CSV.
- Limpieza de artefactos: ignore de logs en .gitignore. Sin .coverage, coverage.xml ni htmlcov en el repo.

## Local QA
poetry check
poetry run ruff check .
poetry run black --check .
poetry run mypy portfolio
PYTHONPATH=. poetry run pytest -q --disable-warnings --maxfail=1 --cov=portfolio --cov-report=term-missing --cov-fail-under=60

## Checklist
- [x] Rebase limpio sobre main
- [x] Lint ok con ruff
- [x] Formato ok con black
- [x] mypy ok en portfolio/*
- [x] Tests verdes con pytest
- [x] Cobertura ≥ 60 %
- [x] Sin artefactos de cobertura ni ficheros generados en repo
- [x] .gitignore actualizado para logs/
- [x] Plots renderizan sin show_plot y sin claves duplicadas

## Merge strategy
Por política del repo: sin merge commits en main.
Usar "Squash and merge" o "Rebase and merge".

## Notes
Si algún check cae por 1–2 puntos de cobertura, tengo preparado un microtest de smoke sobre reporting para subir +1 % sin tocar lógica.
