"""
Test with QQQ and VOO - the exact case that was failing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from portfolio.io.excel_export import export_quant_metrics_to_excel


def test_qqq_voo():
    """Test with QQQ and VOO from 2020-01-01"""
    print("Testing with QQQ and VOO (exact user case)...")
    print("=" * 80)

    try:
        excel_bytes = export_quant_metrics_to_excel(
            tickers=["QQQ", "VOO"],
            start="2020-01-01",
            end="2024-12-31",
            benchmark="SPY",
            rf_annual=0.02,
            vol_lookback=20,
            vol_method="parkinson",
        )

        print("✅ SUCCESS!")
        print(f"Excel size: {len(excel_bytes):,} bytes")

        # Save to file for inspection
        with open("test_qqq_voo.xlsx", "wb") as f:
            f.write(excel_bytes)
        print("✅ Saved to test_qqq_voo.xlsx")

        return True

    except Exception as e:
        print("❌ FAILED!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qqq_voo()
    sys.exit(0 if success else 1)
