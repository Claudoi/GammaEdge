"""
Test that exactly replicates Streamlit's behavior with float vol_lookback
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from portfolio.io.excel_export import export_quant_metrics_to_excel


def test_with_float_lookback():
    """Test with float vol_lookback (as Streamlit st.number_input returns)"""
    print("Testing with FLOAT vol_lookback (Streamlit behavior)...")
    print("=" * 80)

    # Simulate Streamlit st.number_input which returns float
    vol_lookback_from_streamlit = 20.0  # This is what st.number_input returns!

    try:
        excel_bytes = export_quant_metrics_to_excel(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            benchmark="SPY",
            rf_annual=0.02,
            vol_lookback=vol_lookback_from_streamlit,  # FLOAT, not int!
            vol_method="parkinson",
        )

        print("✅ SUCCESS with float lookback!")
        print(f"Size: {len(excel_bytes)} bytes")

    except Exception as e:
        print("❌ FAILED with float lookback!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_with_int_lookback():
    """Test with int vol_lookback (after conversion)"""
    print("\nTesting with INT vol_lookback (after conversion)...")
    print("=" * 80)

    try:
        excel_bytes = export_quant_metrics_to_excel(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            benchmark="SPY",
            rf_annual=0.02,
            vol_lookback=int(20.0),  # Explicitly converted to int
            vol_method="parkinson",
        )

        print("✅ SUCCESS with int lookback!")
        print(f"Size: {len(excel_bytes)} bytes")

    except Exception as e:
        print("❌ FAILED with int lookback!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    result1 = test_with_float_lookback()
    result2 = test_with_int_lookback()

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Float lookback: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"Int lookback: {'✅ PASS' if result2 else '❌ FAIL'}")
