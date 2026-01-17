# Dataset Card: qqq_voo_spy_common_window_yahoo_wide

## Overview

| Property | Value |
|----------|-------|
| **Dataset ID** | `qqq_voo_spy_common_window_yahoo_wide` |
| **Version** | `v1.0.0` |
| **Panel Type** | `custom_window` |
| **Provider** | `yahoo` |
| **Created** | `2026-01-07T13:59:28.330443Z` |
| **Schema Version** | `wide_v1` |

## Description

QQQ/VOO/SPY historical dataset (Wide Format). Mode: common_window.

## What is this dataset?

This is a frozen, auditable dataset containing historical price and feature data
for quantitative finance research and machine learning.

## Instruments

| Ticker | Start Date | End Date | Rows |
|--------|------------|----------|------|
| QQQ | 2010-09-09 | 2023-12-29 | 3,350 |
| VOO | 2010-09-09 | 2023-12-29 | 3,350 |
| SPY | 2010-09-09 | 2023-12-29 | 3,350 |

## Time Range

- **Requested**: 2010-09-09 → 2023-12-31
- **Common Window**: 2010-09-09 → 2023-12-29

## Data Sources

- **Provider**: yahoo
- **Adjustment Version**: 2.0.0
- **Feature Set Version**: v1
- **Calendar**: NYSE

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
| **Status** | `WARN` |
| **Total Rows** | 3,350 |
| **Total Columns** | 43 |
| **Event Days** | 0 |
| **Panel Balanced** | True |

### Issues Detected

- ⚠️ High missingness in dd_20d_BIL: 100.0%
- ⚠️ High missingness in mom_60d_BIL: 100.0%
- ⚠️ High missingness in ret_1d_BIL: 100.0%
- ⚠️ High missingness in ret_20d_BIL: 100.0%
- ⚠️ High missingness in ret_5d_BIL: 100.0%
- ⚠️ High missingness in vol_20d_BIL: 100.0%

## How to Reproduce

```bash
cd GammaEdge
python scripts/freeze_dataset.py \
    --provider yahoo \
    --mode max_history \
    --tickers QQQ VOO SPY
```

## Verification (Audit)

```python
import hashlib
from pathlib import Path

# Verify Truth (Parquet)
p = Path("data.parquet")
actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()

EXPECTED_HASH_FULL = "06d3c3cc1c710c9eac261a603bf526ed766d8222b61dc5363904f74a04eda658"
assert actual_hash == EXPECTED_HASH_FULL, f"Data integrity check failed! Got {actual_hash}"
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
| v1.0.0 | 2026-01-07 | Initial release |

## Hashes (Manifest)

See `RELEASE.json` for full audit details including environment fingerprint.

- **Content (Parquet)**: `06d3c3cc1c710c9e...`
- **Schema**: `bf33cc971643bcfd`
- **Git Commit**: `74c9885c81f0`
