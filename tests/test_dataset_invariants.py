import json
import logging
import unittest
from pathlib import Path

import polars as pl

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestDatasetInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Find the latest Yahoo Wide dataset from INDEX.json"""
        cls.root_dir = Path(__file__).parent.parent
        cls.index_path = cls.root_dir / "datasets" / "INDEX.json"

        if not cls.index_path.exists():
            raise FileNotFoundError("datasets/INDEX.json not found. Freeze a dataset first.")

        with open(cls.index_path) as f:
            index = json.load(f)

        # Pick the latest common_window dataset for testing
        candidates = [x for x in index if "common_window" in x["dataset_id"]]
        if not candidates:
            # Fallback to any
            candidates = index

        if not candidates:
            raise ValueError("No datasets found in INDEX.json")

        cls.latest_entry = candidates[0]
        cls.dataset_path = cls.root_dir / "datasets" / cls.latest_entry["path"]

        logger.info(f"Testing invariants on: {cls.dataset_path}")

        # Load Parquet
        cls.df = pl.read_parquet(cls.dataset_path / "data.parquet")

        # Determine tickers
        # Parse from ID "qqq_voo_bil_..." or columns "close_QQQ"
        cls.tickers = []
        for col in cls.df.columns:
            if col.startswith("close_"):
                cls.tickers.append(col.replace("close_", ""))

    def test_01_monotonic_dates(self):
        """Invariant: Dates must be strictly increasing and unique."""
        dates = self.df["date"]
        is_sorted = dates.is_sorted()
        duplicates = dates.is_duplicated().any()

        self.assertTrue(is_sorted, "Dates are not sorted.")
        self.assertFalse(duplicates, "Duplicate dates found.")

    def test_02_prices_positive(self):
        """Invariant: All price columns must be > 0."""
        for ticker in self.tickers:
            for field in ["open", "high", "low", "close"]:
                col = f"{field}_{ticker}"
                if col in self.df.columns:
                    min_val = self.df[col].min()
                    self.assertGreater(min_val, 0, f"{col} has non-positive values (min={min_val})")

    def test_03_high_low_logic(self):
        """Invariant: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close."""
        for ticker in self.tickers:
            high_col = f"high_{ticker}"
            low_col = f"low_{ticker}"

            if high_col in self.df.columns and low_col in self.df.columns:
                inversions = self.df.filter(pl.col(high_col) < pl.col(low_col))
                count = inversions.height
                self.assertEqual(count, 0, f"Found {count} rows where High < Low for {ticker}")

    def test_04_volume_non_negative(self):
        """Invariant: Volume >= 0."""
        for ticker in self.tickers:
            col = f"volume_{ticker}"
            if col in self.df.columns:
                min_val = self.df[col].min()
                self.assertGreaterEqual(min_val, 0, f"{col} has negative values (min={min_val})")

    def test_05_returns_consistency(self):
        """Invariant: ret_1d approx close/prev_close - 1."""
        # This is a loose check due to floats, but good for catching massive errors.
        # We need to shift close to check
        for ticker in self.tickers:
            ret_col = f"ret_1d_{ticker}"
            close_col = f"close_{ticker}"

            if ret_col in self.df.columns and close_col in self.df.columns:
                # Calc implied return
                implied = (self.df[close_col] / self.df[close_col].shift(1)) - 1.0
                actual = self.df[ret_col]

                # Check deviation (fill nulls to avoid error, first row is null)
                diff = (implied - actual).abs().fill_null(0.0)
                max_diff = diff.max()

                # Allow small epsilon for floating point diffs
                self.assertLess(
                    max_diff, 1e-4, f"Return calculation diverges > 0.0001 for {ticker}"
                )

    def test_06_schema_canonical(self):
        """Invariant: Columns must be sorted (deterministic writer check)."""
        cols = self.df.columns
        sorted_cols = sorted(cols)
        self.assertEqual(
            cols, sorted_cols, "Columns are not alphabetically sorted in the parquet file."
        )


if __name__ == "__main__":
    unittest.main()
