"""
Phase 1 Audit Automated Tests
Validates the data integrity, dimensions, target distribution, and audit outputs.
Executable via unittest or pytest.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, get_raw_data_path
from src.data_audit import run_data_audit


class TestPhase1DataAudit(unittest.TestCase):
    def test_raw_data_exists(self):
        """Verify raw dataset file exists and is non-empty."""
        path = get_raw_data_path()
        self.assertTrue(path.exists(), f"Raw dataset not found at {path}")
        self.assertGreater(path.stat().st_size, 0, "Raw dataset file is empty")

    def test_raw_data_dimensions(self):
        """Verify raw dataset has expected dimensions (9240 rows, 37 columns)."""
        df = load_raw_data()
        self.assertEqual(df.shape, (9240, 37), f"Unexpected shape: {df.shape}")

    def test_target_integrity(self):
        """Verify target column 'Converted' is binary, non-null, and correctly distributed."""
        df = load_raw_data()
        self.assertIn("Converted", df.columns, "Target column 'Converted' is missing")
        self.assertEqual(df["Converted"].isna().sum(), 0, "Target column contains null values")

        unique_vals = set(df["Converted"].unique())
        self.assertEqual(unique_vals, {0, 1}, f"Target has invalid values: {unique_vals}")

        counts = df["Converted"].value_counts()
        self.assertEqual(counts[0], 5679, f"Expected 5679 non-converted, got {counts[0]}")
        self.assertEqual(counts[1], 3561, f"Expected 3561 converted, got {counts[1]}")

    def test_id_uniqueness(self):
        """Verify Prospect ID and Lead Number are 100% unique without duplicates."""
        df = load_raw_data()
        self.assertEqual(df["Prospect ID"].nunique(), 9240, "Duplicate Prospect IDs found")
        self.assertEqual(df["Lead Number"].nunique(), 9240, "Duplicate Lead Numbers found")
        self.assertEqual(df.duplicated().sum(), 0, "Duplicate full rows found")

    def test_audit_artifacts_generated(self):
        """Verify missing values CSV report was generated with 37 rows and valid columns."""
        run_data_audit()
        report_path = PROJECT_ROOT / "reports" / "missing_values_report.csv"
        self.assertTrue(report_path.exists(), "missing_values_report.csv does not exist")

        report_df = pd.read_csv(report_path)
        self.assertEqual(len(report_df), 37, f"Expected 37 columns in report, got {len(report_df)}")
        self.assertIn("Total_Missing_Or_Select_Pct", report_df.columns)


if __name__ == "__main__":
    unittest.main()

