"""
Diagnostic script to reproduce and debug the Excel export error.
Run this to get the full traceback.
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from portfolio.io.excel_export import export_quant_metrics_to_excel

def test_export():
    """Test the export with minimal data to isolate the error."""
    try:
        print("Starting Excel export test...")
        print("=" * 80)
        
        excel_bytes = export_quant_metrics_to_excel(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            benchmark="SPY",
            rf_annual=0.02,
            vol_lookback=20,
            vol_method="parkinson",
            output_format="bytes",
        )
        
        print("✅ SUCCESS! Excel generated successfully")
        print(f"Size: {len(excel_bytes)} bytes")
        
    except Exception as e:
        print("❌ ERROR OCCURRED:")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        print("-" * 80)
        traceback.print_exc()
        print("=" * 80)
        
        # Additional debugging info
        print("\nDebugging Information:")
        print(f"Python version: {sys.version}")
        
        import polars as pl
        print(f"Polars version: {pl.__version__}")
        
        return False
    
    return True

if __name__ == "__main__":
    success = test_export()
    sys.exit(0 if success else 1)
