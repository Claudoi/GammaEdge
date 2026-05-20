# Dataset Freeze Infrastructure
# =============================
"""
Infraestructura para congelar datasets de forma profesional.

Cada dataset es una unidad autocontenida:

datasets/
└── {dataset_id}_v{version}/
    ├── data.parquet              # fuente de verdad
    ├── data.xlsx                 # vista humana
    ├── metadata.json             # metadata estructurada
    ├── dataset_card.md           # documentación humana
    └── quality_report.json       # certificado de sanidad

REGLAS:
- Parquet es TRUTH, Excel es PRESENTATION
- Cualquier cambio → nueva versión
- Hash como contrato
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Definition
# =============================================================================


@dataclass
class DatasetDefinition:
    """Definición de un dataset frozen."""

    # Identity
    dataset_id: str
    dataset_version: str = "v1.0.0"
    panel_type: Literal["balanced", "unbalanced"] = "balanced"
    description: str = ""

    # Source
    provider: str = "massive"
    tickers: list[str] = field(default_factory=list)

    # Time Range
    requested_start: date | None = None
    requested_end: date | None = None

    # Versions
    adjustment_version: str = "2.0.0"
    feature_set_version: str = "v1"
    calendar_id: str = "NYSE"

    def get_folder_name(self) -> str:
        """Nombre de la carpeta del dataset."""
        return f"{self.dataset_id}_{self.dataset_version}"


# =============================================================================
# Quality Report Generator
# =============================================================================


class QualityReportGenerator:
    """Genera quality report para un dataset."""

    def __init__(self, df: pl.DataFrame, definition: DatasetDefinition):
        self.df = df
        self.definition = definition

    def generate(self) -> dict:
        """Genera el quality report completo."""
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "dataset_id": self.definition.dataset_id,
            "dataset_version": self.definition.dataset_version,
        }

        # Coverage
        report["coverage"] = self._check_coverage()

        # Missingness
        report["missingness"] = self._check_missingness()

        # Quality flags
        report["quality_flags"] = self._check_quality_flags()

        # Event days
        report["event_days"] = self._check_event_days()

        # Sanity checks
        report["sanity_checks"] = self._run_sanity_checks()

        # Panel balance
        report["panel_balance"] = self._check_panel_balance()

        # Summary
        report["summary"] = self._create_summary(report)

        return report

    def _check_coverage(self) -> dict:
        """Verifica coverage por ticker."""
        coverage = {}

        # Detect Wide
        is_wide = "ticker" not in self.df.columns

        if is_wide:
            for ticker in self.definition.tickers:
                # Check for close_{ticker} or similar
                col_name = f"close_{ticker}"
                if col_name not in self.df.columns:
                    col_name = f"adj_close_{ticker}"  # Try another

                if col_name in self.df.columns:
                    # Count non-nulls
                    valid_rows = self.df[col_name].drop_nulls().len()

                    if valid_rows == 0:
                        coverage[ticker] = {"status": "MISSING", "rows": 0}
                    else:
                        start = str(self.df.filter(pl.col(col_name).is_not_null())["date"].min())
                        end = str(self.df.filter(pl.col(col_name).is_not_null())["date"].max())
                        coverage[ticker] = {
                            "status": "OK",
                            "rows": valid_rows,
                            "start": start,
                            "end": end,
                        }
                else:
                    coverage[ticker] = {"status": "COLUMN_MISSING", "rows": 0}
        else:
            for ticker in self.definition.tickers:
                df_ticker = self.df.filter(pl.col("ticker") == ticker)

                if df_ticker.height == 0:
                    coverage[ticker] = {"status": "MISSING", "rows": 0}
                    continue

                coverage[ticker] = {
                    "status": "OK",
                    "rows": df_ticker.height,
                    "start": str(df_ticker["date"].min()),
                    "end": str(df_ticker["date"].max()),
                }

        return coverage

    def _check_missingness(self) -> dict:
        """Verifica valores faltantes por columna."""
        missingness = {}

        for col in self.df.columns:
            null_count = self.df[col].null_count()
            total = self.df.height
            pct = null_count / total if total > 0 else 0

            missingness[col] = {
                "null_count": null_count,
                "pct": round(pct * 100, 2),
            }

        return missingness

    def _check_quality_flags(self) -> dict:
        """Verifica quality flags."""
        if "quality_flag" not in self.df.columns:
            return {"status": "NO_QUALITY_FLAG_COLUMN"}

        counts = self.df.group_by("quality_flag").count().to_dicts()
        return {row["quality_flag"]: row["count"] for row in counts}

    def _check_event_days(self) -> dict:
        """Verifica event days (splits, dividends)."""
        res = {"count": 0, "dividend_events": 0, "split_events": 0, "status": "OK"}

        # Check general is_event_day
        if "is_event_day" in self.df.columns:
            res["count"] = self.df.filter(pl.col("is_event_day")).height

        # Check specific columns if they exist (from wide format)
        # Often wide format might not carry explicit dividend/split columns for all tickers
        # But if we had them or if we check features...

        # For now, rely on "is_event_day" which is the aggregated flag

        return res

    def _run_sanity_checks(self) -> dict:
        """Ejecuta sanity checks en los datos."""
        checks = {}

        is_wide = "ticker" not in self.df.columns

        if is_wide:
            # Check main ticker (QQQ) or first one
            main_ticker = self.definition.tickers[0] if self.definition.tickers else "QQQ"
            close_col = f"close_{main_ticker}"
            if close_col in self.df.columns:
                checks["close_min"] = float(self.df[close_col].min() or 0)
                checks["close_max"] = float(self.df[close_col].max() or 0)
                checks["close_negative"] = int(self.df.filter(pl.col(close_col) < 0).height)
        else:
            # Price checks
            if "close" in self.df.columns:
                checks["close_min"] = float(self.df["close"].min() or 0)
                checks["close_max"] = float(self.df["close"].max() or 0)
                checks["close_negative"] = int(self.df.filter(pl.col("close") < 0).height)

        return checks

    def _check_panel_balance(self) -> dict:
        """Verifica si el panel está balanceado."""
        rows_by_ticker = {}

        is_wide = "ticker" not in self.df.columns
        if is_wide:
            # Check non-nulls per close column
            for ticker in self.definition.tickers:
                col = f"close_{ticker}"
                if col in self.df.columns:
                    rows_by_ticker[ticker] = self.df[col].drop_nulls().len()
                else:
                    rows_by_ticker[ticker] = 0
        else:
            for ticker in self.definition.tickers:
                rows_by_ticker[ticker] = self.df.filter(pl.col("ticker") == ticker).height

        unique_counts = set(rows_by_ticker.values())
        is_balanced = len(unique_counts) == 1

        return {
            "is_balanced": is_balanced,
            "rows_by_ticker": rows_by_ticker,
        }

    def _create_summary(self, report: dict) -> dict:
        """Crea resumen del quality report."""
        issues = []

        # Check for missing data
        for col, miss in report["missingness"].items():
            if miss["pct"] > 5:
                issues.append(f"High missingness in {col}: {miss['pct']}%")

        # Check for suspect rows
        if isinstance(report["quality_flags"], dict):
            suspect = report["quality_flags"].get("suspect", 0)
            if suspect > 0:
                issues.append(f"Suspect rows: {suspect}")

        # Check panel balance
        if not report["panel_balance"]["is_balanced"]:
            issues.append("Panel is unbalanced")

        return {
            "status": "PASS" if not issues else "WARN",
            "issues": issues,
            "total_rows": self.df.height,
            "total_columns": len(self.df.columns),
        }


# =============================================================================
# Dataset Card Generator
# =============================================================================


class DatasetCardGenerator:
    """Genera dataset card en markdown."""

    def __init__(
        self,
        definition: DatasetDefinition,
        metadata: dict,
        quality_report: dict,
        manifest: dict = None,
    ):
        self.definition = definition
        self.metadata = metadata
        self.quality_report = quality_report
        self.manifest = manifest or {}

    def generate(self) -> str:
        """Genera el contenido del dataset card."""
        d = self.definition
        m = self.metadata
        q = self.quality_report
        man = self.manifest

        card = f"""# Dataset Card: {d.dataset_id}

## Overview

| Property | Value |
|----------|-------|
| **Dataset ID** | `{d.dataset_id}` |
| **Version** | `{d.dataset_version}` |
| **Panel Type** | `{d.panel_type}` |
| **Provider** | `{d.provider}` |
| **Created** | `{m.get('created_at', 'unknown')}` |
| **Schema Version** | `{m.get('schema_version', 'wide_v1')}` |

## Description

{d.description or 'No description provided.'}

## What is this dataset?

This is a frozen, auditable dataset containing historical price and feature data
for quantitative finance research and machine learning.

## Instruments

| Ticker | Start Date | End Date | Rows |
|--------|------------|----------|------|
"""
        # Add ticker rows
        for ticker in d.tickers:
            coverage = q.get("coverage", {}).get(ticker, {})
            start = coverage.get("start", "N/A")
            end = coverage.get("end", "N/A")
            rows = coverage.get("rows", 0)
            card += f"| {ticker} | {start} | {end} | {rows:,} |\n"

        card += f"""
## Time Range

- **Requested**: {m.get('requested_start', 'N/A')} → {m.get('requested_end', 'N/A')}
- **Common Window**: {m.get('common_start', 'N/A')} → {m.get('common_end', 'N/A')}

## Data Sources

- **Provider**: {d.provider}
- **Adjustment Version**: {d.adjustment_version}
- **Feature Set Version**: {d.feature_set_version}
- **Calendar**: {d.calendar_id}

## What adjustments were applied?

1. **Corporate Actions**: Splits and dividends adjusted using factor-based method
2. **Volume**: Inversely adjusted for splits (dollar volume preserved)
3. **Quality Gates**: Automatic detection of suspect data points

## Columns Included (WIDE Format)

### Identity
- `date`: Trading date in exchange calendar, timezone-normalized

### Price Data (per asset)
e.g. `close_QQQ`, `volume_VOO`
- `open_[TICKER]`, `high_[TICKER]`, `low_[TICKER]`, `close_[TICKER]`: Adjusted prices in USD
- `volume_[TICKER]`: Adjusted volume (shares) to preserve dollar volume

### Features (per asset)
e.g. `ret_1d_BIL`
- `ret_1d`: 1-day simple return (close-to-close)
- `ret_5d`, `ret_20d`: Rolling simple returns
- `realized_vol_20d`: Annualized volatility (20d)
- `momentum_12_1`: 12-month momentum excluding last month
- `drawdown_20d`: Drawdown from 20d high
- `dollar_volume`: close * volume

### Quality
- `quality_flag`: "OK" or warning label

## What is NOT included?

❌ Forward returns (labels)
❌ Strategy-specific features (beyond basics)
❌ Raw unadjusted prices (available separately in data_lake)

## Limitations

- Panel may be unbalanced if using `max_history` mode
- ETF data only (no individual stocks)
- US markets only (NYSE/NASDAQ)

## Quality Report Summary

| Metric | Value |
|--------|-------|
| **Status** | `{q.get('summary', {}).get('status', 'UNKNOWN')}` |
| **Total Rows** | {q.get('summary', {}).get('total_rows', 0):,} |
| **Total Columns** | {q.get('summary', {}).get('total_columns', 0)} |
| **Event Days** | {q.get('event_days', {}).get('count', 0)} |
| **Panel Balanced** | {q.get('panel_balance', {}).get('is_balanced', False)} |

"""
        # Add issues if any
        issues = q.get("summary", {}).get("issues", [])
        if issues:
            card += "### Issues Detected\n\n"
            for issue in issues:
                card += f"- ⚠️ {issue}\n"
            card += "\n"

        content_hash = man.get("hashes", {}).get("content_full", "unknown")

        card += f"""## How to Reproduce

```bash
cd GammaEdge
python scripts/freeze_dataset.py \\
    --provider {d.provider} \\
    --mode {'common_window' if d.panel_type == 'balanced' else 'max_history'} \\
    --tickers {' '.join(d.tickers)}
```

## Verification (Audit)

```python
import hashlib
from pathlib import Path

# Verify Truth (Parquet)
p = Path("data.parquet")
actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()

EXPECTED_HASH_FULL = "{content_hash}"
assert actual_hash == EXPECTED_HASH_FULL, f"Data integrity check failed! Got {{actual_hash}}"
print("✅ Data integrity verified (Full SHA-256).")
```

## Usage Rules

❌ Do NOT modify the Excel file manually
❌ Do NOT add columns manually
❌ Do NOT recalculate features in consumer projects
❌ Do NOT mix datasets of different versions

✅ If anything changes → create new dataset version
✅ ML projects = read-only consumers

## Version History

| Version | Date | Changes |
|---------|------|---------|
| {d.dataset_version} | {m.get('created_at', 'unknown')[:10]} | Initial release |

## Hashes (Manifest)

See `RELEASE.json` for full audit details including environment fingerprint.

- **Content (Parquet)**: `{content_hash[:16]}...`
- **Schema**: `{m.get('schema_hash', 'unknown')}`
- **Git Commit**: `{m.get('git_commit', 'unknown')}`
"""

        return card


# =============================================================================
# Dataset Freezer (main class)
# =============================================================================


class DatasetFreezer:
    """
    Congela un dataset como unidad autocontenida.

    Genera:
    - data.parquet (truth)
    - data.xlsx (presentation)
    - metadata.json
    - dataset_card.md
    - quality_report.json
    """

    def __init__(self, base_path: str | Path = "datasets"):
        self.base_path = Path(base_path)

    def freeze(
        self,
        df: pl.DataFrame,
        definition: DatasetDefinition,
    ) -> Path:
        """
        Congela el dataset.

        Returns:
            Path to the frozen dataset folder
        """
        # Create folder
        folder = self.base_path / definition.get_folder_name()
        folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Freezing dataset to {folder}")

        # 1. Deterministic Prep
        # - Sort columns alphabetically
        # - Sort rows by date (and PK if present)
        # - Ensure float32 vs float64 consistency

        # Sort columns
        cols = sorted(df.columns)
        df = df.select(cols)

        # Sort rows
        # If wide, sort bytes date
        if "date" in df.columns:
            df = df.sort("date")

        # 2. Write Parquet (Truth)
        parquet_path = folder / "data.parquet"

        # Use strict options for reproducibility
        df.write_parquet(
            parquet_path,
            compression="snappy",
            statistics=True,
            use_pyarrow=False,  # Use native polars writer for consistency if possible, or force pyarrow with fixed version
            # row_group_size is handled by specific polars/arrow logic, usually safe default
        )
        logger.info(f"  Written: {parquet_path}")

        # 2. Compute Robust Hashes (Full + Short)
        content_hash_full = self._compute_file_hash(parquet_path)
        content_hash_short = content_hash_full[:16]
        logger.info(f"  Content hash (full): {content_hash_full}")
        logger.info(f"  Content hash (short): {content_hash_short}")

        # 3. Get Env & Git
        git_commit = self._get_git_commit()
        env_fingerprint = self._get_env_fingerprint()

        # 4. Generate quality report
        quality_gen = QualityReportGenerator(df, definition)
        quality_report = quality_gen.generate()

        # 5. Build metadata (Partial)
        metadata = self._build_metadata(df, definition, content_hash_short, git_commit)
        metadata["schema_version"] = "wide_v1"

        # 6. Write Excel & Hash it
        excel_path = folder / "data.xlsx"
        self._write_excel(df, excel_path, metadata, quality_report)
        logger.info(f"  Written: {excel_path}")

        excel_hash_full = self._compute_file_hash(excel_path)
        metadata["excel_hash"] = excel_hash_full[:16]

        # 7. Write Metadata & Hash it
        metadata_path = folder / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"  Written: {metadata_path}")

        metadata_hash_full = self._compute_file_hash(metadata_path)

        # 8. Create RELEASE.json (Manifest)
        release_manifest = {
            "dataset_id": definition.dataset_id,
            "dataset_version": definition.dataset_version,
            "created_at": metadata["created_at"],
            "provider": definition.provider,
            "hashes": {
                "content_full": content_hash_full,
                "content": content_hash_short,
                "excel_full": excel_hash_full,
                "excel": excel_hash_full[:16],
                "metadata_full": metadata_hash_full,
                "metadata": metadata_hash_full[:16],
                "schema": metadata["schema_hash"],
            },
            "environment": env_fingerprint,
            "git_commit": git_commit,
            "adjustment_version": definition.adjustment_version,
            "feature_set_version": definition.feature_set_version,
            "schema_version": metadata["schema_version"],
            # Signature block (metadata)
            "signature": {
                "type": "openssh-ed25519",
                "namespace": "gammaedge-dataset-release",
                "key_id": "gammaedge-release-ed25519",
                "file": "RELEASE.sig",
            },
        }

        release_path = folder / "RELEASE.json"

        # Write Deterministic JSON (sort_keys=True, separators)
        with open(release_path, "w") as f:
            json.dump(
                release_manifest,
                f,
                indent=2,
                default=str,
                sort_keys=True,
                # Use default separators implies (', ', ': ') which is fine,
                # but explicit separators=(',', ':') is more canonical.
                # However, for human readability indent=2 is preferred.
                # We stick to indent=2 + sort_keys=True as our canonical format on disk.
            )
        logger.info(f"  Written: {release_path}")

        # 9. Sign RELEASE.json
        self._sign_manifest(release_path)

        # 10. Generate dataset card
        card_gen = DatasetCardGenerator(definition, metadata, quality_report, release_manifest)
        dataset_card = card_gen.generate()

        # 11. Write remaining files

        # Quality report
        quality_path = folder / "quality_report.json"
        with open(quality_path, "w") as f:
            json.dump(quality_report, f, indent=2)
        logger.info(f"  Written: {quality_path}")

        # Dataset card
        card_path = folder / "dataset_card.md"
        with open(card_path, "w") as f:
            f.write(dataset_card)
        logger.info(f"  Written: {card_path}")

        # 12. Update Global Index
        self._update_global_index(release_manifest, folder, content_hash_full)

        logger.info(f"✅ Dataset frozen: {folder}")
        logger.info(f"   Manifest: {release_path}")
        logger.info(f"   Signature: {folder / 'RELEASE.sig'}")

        return folder

    def _sign_manifest(self, manifest_path: Path):
        """Firma el manifest usando ssh-keygen."""
        key_path = Path("keys/gammaedge_release_ed25519")
        if not key_path.exists():
            logger.warning(f"⚠️ Signing key not found at {key_path}. Skipping signature.")
            return

        namespace = "gammaedge-dataset-release"

        try:
            # ssh-keygen -Y sign -f key -n namespace file
            cmd = [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(key_path),
                "-n",
                namespace,
                str(manifest_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Outcome is manifest_path + ".sig"
            sig_source = manifest_path.with_name(manifest_path.name + ".sig")
            sig_dest = manifest_path.with_name("RELEASE.sig")

            if sig_source.exists():
                sig_source.rename(sig_dest)
                logger.info(f"  Signed: {sig_dest}")

                # Copy public key too for portability
                pub_key = key_path.with_suffix(".pub")
                if pub_key.exists():
                    import shutil

                    shutil.copy(pub_key, manifest_path.parent / "SIGNING.pub")
            else:
                logger.error("  Signature file not created by ssh-keygen")

        except subprocess.CalledProcessError as e:
            logger.error(f"  Signing failed: {e.stderr}")

    def _compute_file_hash(self, path: Path) -> str:
        """Computa hash SHA-256 (64 chars)."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

    def _get_env_fingerprint(self) -> dict:
        """Captura versiones del entorno."""
        import platform

        import polars

        fingerprint = {
            "python": platform.python_version(),
            "polars": polars.__version__,
            "os": platform.platform(),
        }
        try:
            import yfinance

            fingerprint["yfinance"] = yfinance.__version__
        except ImportError:
            pass
        return fingerprint

    def _update_global_index(self, manifest: dict, folder: Path, content_hash_full: str):
        """Actualiza datasets/INDEX.json."""
        index_path = self.base_path / "INDEX.json"
        index = []
        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
            except Exception:
                pass

        # Remove existing entry
        index = [
            i
            for i in index
            if not (
                i["dataset_id"] == manifest["dataset_id"]
                and i["dataset_version"] == manifest["dataset_version"]
            )
        ]

        entry = {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "provider": manifest["provider"],
            "created_at": manifest["created_at"],
            "path": str(folder.name),
            "content_hash_short": manifest["hashes"]["content"],
            "content_hash_full": content_hash_full,
        }
        index.append(entry)
        index.sort(key=lambda x: x["created_at"], reverse=True)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"  Updated Index: {index_path}")

    def _compute_hash(self, df: pl.DataFrame) -> str:
        """Deprecated."""
        return "DEPRECATED"

    def _get_git_commit(self) -> str:
        """Obtiene el commit actual de git."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    def _build_metadata(
        self,
        df: pl.DataFrame,
        definition: DatasetDefinition,
        content_hash: str,
        git_commit: str,
    ) -> dict:
        """Construye metadata completa."""
        # Stats logic...
        actual_start_by_ticker = {}
        actual_end_by_ticker = {}
        rows_by_ticker = {}

        # Detect if Wide or Long
        is_wide = "ticker" not in df.columns

        if not is_wide:
            for ticker in definition.tickers:
                df_ticker = df.filter(pl.col("ticker") == ticker)
                if df_ticker.height > 0:
                    actual_start_by_ticker[ticker] = str(df_ticker["date"].min())
                    actual_end_by_ticker[ticker] = str(df_ticker["date"].max())
                    rows_by_ticker[ticker] = df_ticker.height
        else:
            # For Wide, assume all tickers cover the date range present (simplified)
            # Or check specific columns if needed.
            min_date = str(df["date"].min())
            max_date = str(df["date"].max())
            for ticker in definition.tickers:
                actual_start_by_ticker[ticker] = min_date
                actual_end_by_ticker[ticker] = max_date
                rows_by_ticker[ticker] = df.height

        # Common window
        if actual_start_by_ticker:
            common_start = max(actual_start_by_ticker.values())
            common_end = min(actual_end_by_ticker.values())
        else:
            common_start = common_end = None

        # Schema hash
        schema_info = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes, strict=False)}
        schema_hash = hashlib.sha256(json.dumps(schema_info, sort_keys=True).encode()).hexdigest()[
            :16
        ]

        return {
            "dataset_id": definition.dataset_id,
            "dataset_version": definition.dataset_version,
            "panel_type": definition.panel_type,
            "description": definition.description,
            "provider": definition.provider,
            "tickers": definition.tickers,
            "requested_start": (
                str(definition.requested_start) if definition.requested_start else None
            ),
            "requested_end": str(definition.requested_end) if definition.requested_end else None,
            "actual_start_by_ticker": actual_start_by_ticker,
            "actual_end_by_ticker": actual_end_by_ticker,
            "common_start": common_start,
            "common_end": common_end,
            "rows_by_ticker": rows_by_ticker,
            "total_rows": df.height,
            "total_columns": len(df.columns),
            "columns": df.columns,
            "adjustment_version": definition.adjustment_version,
            "feature_set_version": definition.feature_set_version,
            "calendar_id": definition.calendar_id,
            "content_hash": content_hash,
            "schema_hash": schema_hash,
            "git_commit": git_commit,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "gammaedge_version": "1.0.0",
        }

    def _write_excel(self, df: pl.DataFrame, path: Path, metadata: dict, quality_report: dict):
        """Escribe Excel con Sheets: DATA, METADATA, DICTIONARY, QUALITY."""
        try:
            import xlsxwriter

            workbook = xlsxwriter.Workbook(str(path))

            # Formats
            header_fmt = workbook.add_format(
                {"bold": True, "bg_color": "#1a1a2e", "font_color": "white", "border": 1}
            )

            # 1. DATA
            ws_data = workbook.add_worksheet("DATA")
            for i, col in enumerate(df.columns):
                ws_data.write(0, i, col, header_fmt)

            # Write data (optimized loop)
            for r, row in enumerate(df.iter_rows(named=True), 1):
                for c, col in enumerate(df.columns):
                    val = row[col]
                    if val is None:
                        continue
                    if isinstance(val, (date, datetime)):
                        ws_data.write(r, c, str(val))
                    else:
                        ws_data.write(r, c, val)
            ws_data.freeze_panes(1, 0)

            # 2. METADATA
            ws_meta = workbook.add_worksheet("METADATA")
            ws_meta.write(0, 0, "Key", header_fmt)
            ws_meta.write(0, 1, "Value", header_fmt)
            for r, (k, v) in enumerate(metadata.items(), 1):
                ws_meta.write(r, 0, k)
                ws_meta.write(
                    r, 1, json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
                )
            ws_meta.set_column(0, 0, 25)
            ws_meta.set_column(1, 1, 80)

            # 3. DATA DICTIONARY
            ws_dict = workbook.add_worksheet("DATA_DICTIONARY")
            dict_headers = ["Column", "Type", "Description", "Lookback"]
            for i, h in enumerate(dict_headers):
                ws_dict.write(0, i, h, header_fmt)

            # Simple dictionary generation
            row_idx = 1
            for col in df.columns:
                ws_dict.write(row_idx, 0, col)
                ws_dict.write(row_idx, 1, str(df.schema[col]))

                desc = (
                    "Raw price" if col in ["open", "high", "low", "close", "volume"] else "Feature"
                )
                if "ret_" in col:
                    desc = "Return"
                if "vol_" in col:
                    desc = "Volatility"
                if "date" in col:
                    desc = "Date"

                ws_dict.write(row_idx, 2, desc)
                row_idx += 1
            ws_dict.set_column(0, 0, 25)

            # 4. QUALITY REPORT
            ws_qual = workbook.add_worksheet("QUALITY_REPORT")
            ws_qual.write(0, 0, "Metric", header_fmt)
            ws_qual.write(0, 1, "Value", header_fmt)

            row_idx = 1

            # Flatten quality report for excel
            def write_quality_item(key, val, indent=0):
                nonlocal row_idx
                prefix = "  " * indent
                if isinstance(val, dict):
                    ws_qual.write(row_idx, 0, f"{prefix}{key}")
                    row_idx += 1
                    for k, v in val.items():
                        write_quality_item(k, v, indent + 1)
                else:
                    ws_qual.write(row_idx, 0, f"{prefix}{key}")
                    ws_qual.write(row_idx, 1, str(val))
                    row_idx += 1

            for k, v in quality_report.items():
                write_quality_item(k, v)

            ws_qual.set_column(0, 0, 40)
            ws_qual.set_column(1, 1, 40)

            workbook.close()

        except ImportError:
            df.write_excel(str(path))

    def verify(self, folder: Path | str) -> bool:
        """Verifica integridad de un dataset frozen (usando bytes)."""
        folder = Path(folder)

        # Load metadata
        metadata_path = folder / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        expected_hash = metadata["content_hash"]

        # Compute hash of file
        parquet_path = folder / "data.parquet"
        if not parquet_path.exists():
            logger.error("data.parquet not found")
            return False

        actual_hash_full = self._compute_file_hash(parquet_path)
        actual_hash_short = actual_hash_full[:16]

        # Check against full if expected is 64 chars, else short
        if len(expected_hash) == 64:
            if actual_hash_full != expected_hash:
                logger.error(f"Hash mismatch! Expected {expected_hash}, got {actual_hash_full}")
                return False
        else:
            if actual_hash_short != expected_hash:
                logger.error(f"Hash mismatch! Expected {expected_hash}, got {actual_hash_short}")
                return False

        logger.info(f"✅ Dataset verified (byte-level): {folder}")
        return True
