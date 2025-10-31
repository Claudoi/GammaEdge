## Local QA
poetry check
poetry run ruff check .
poetry run ruff format .
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
- [x] Sin artefactos de coverage en repo
- [x] .gitignore incluye logs/
- [x] Scenarios: plots sin show_plot y sin claves duplicadas
- [x] CI: py310 dataclass_transform y matriz 3.10/3.11

## Merge strategy
Sin merge commits en main. Usar Squash and merge o Rebase and merge.

## Notes
Si cae cobertura por 1–2 pts, existe micro-smoke test en reporting para +1 % sin tocar lógica.
