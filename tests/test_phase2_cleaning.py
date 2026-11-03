"""
Phase 2 Data Cleaning & EDA Automated Tests
Validates the cleaning transformations, outlier handling, categorical standardization,
and EDA artifact generation.
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

from src.data_loader import get_raw_data_path
from src.data_cleaning import clean_leads_data, load_cleaned_data, get_processed_data_path
from src.eda import calculate_eda_metrics, ASSETS_DIR


class TestPhase2CleaningAndEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure cleaned data is generated for testing."""
        cls.raw_path = get_raw_data_path()
        cls.raw_size = cls.raw_path.stat().st_size
        cls.df_clean = load_cleaned_data()

    def test_raw_data_remains_untouched(self):
        """Verify the raw CSV has not been modified or deleted."""
        self.assertTrue(self.raw_path.exists())
        self.assertEqual(self.raw_path.stat().st_size, self.raw_size)

    def test_cleaned_dataset_row_preservation(self):
        """Verify no valid customer records were discarded (exactly 9,240 rows and 14 columns)."""
        self.assertEqual(len(self.df_clean), 9240)
        self.assertEqual(self.df_clean.shape[1], 14)
        self.assertIn("City", self.df_clean.columns)
        self.assertIn("Lead Quality", self.df_clean.columns)

    def test_no_null_values_in_cleaned_dataset(self):
        """Verify zero missing values across all columns in cleaned data."""
        null_counts = self.df_clean.isna().sum().sum()
        self.assertEqual(null_counts, 0, f"Found {null_counts} null values in cleaned data")

    def test_no_select_placeholders_in_categoricals(self):
        """Verify no lingering 'Select' placeholder strings exist."""
        for col in self.df_clean.select_dtypes(include=["object", "string", "str"]).columns:
            has_select = (self.df_clean[col].astype(str).str.strip().str.lower() == "select").any()
            self.assertFalse(has_select, f"Column '{col}' still contains 'Select' placeholder")

    def test_outlier_capping_total_visits(self):
        """Verify TotalVisits is non-negative and capped at 99th percentile (<= 17.0)."""
        self.assertGreaterEqual(self.df_clean["TotalVisits"].min(), 0.0)
        self.assertLessEqual(self.df_clean["TotalVisits"].max(), 17.0)

    def test_outlier_capping_page_views(self):
        """Verify Page Views Per Visit is non-negative and capped at 99th percentile (<= 9.0)."""
        self.assertGreaterEqual(self.df_clean["Page Views Per Visit"].min(), 0.0)
        self.assertLessEqual(self.df_clean["Page Views Per Visit"].max(), 9.0)

    def test_time_spent_validity(self):
        """Verify Total Time Spent on Website is non-negative and within reasonable bounds."""
        self.assertGreaterEqual(self.df_clean["Total Time Spent on Website"].min(), 0)
        self.assertLessEqual(self.df_clean["Total Time Spent on Website"].max(), 2500)

    def test_source_standardization(self):
        """Verify Lead Source casing is standardized and rare sources collapsed."""
        unique_sources = set(self.df_clean["Lead Source"].unique())
        self.assertNotIn("google", unique_sources, "Lower-case 'google' found in Lead Source")
        self.assertNotIn("facebook", unique_sources, "Lower-case 'facebook' found in Lead Source")
        self.assertIn("Other_Source", unique_sources)

    def test_target_distribution_preserved(self):
        """Verify target column Converted maintains exact counts."""
        counts = self.df_clean["Converted"].value_counts()
        self.assertEqual(counts[0], 5679)
        self.assertEqual(counts[1], 3561)

    def test_eda_metrics_consistency(self):
        """Verify calculate_eda_metrics returns consistent mathematical figures."""
        m = calculate_eda_metrics(self.df_clean)
        self.assertEqual(m["total_leads"], 9240)
        self.assertEqual(m["converted_leads"], 3561)
        self.assertEqual(m["overall_conv_rate"], 38.54)

    def test_eda_chart_assets_exist(self):
        """Verify all 7 EDA chart png files were generated and are non-empty."""
        expected_figs = [
            "fig1_target_distribution.png",
            "fig2_conversion_by_origin.png",
            "fig3_conversion_by_source.png",
            "fig4_conversion_by_occupation.png",
            "fig5_conversion_by_specialization.png",
            "fig6_numerical_distributions.png",
            "fig7_correlation_heatmap.png",
        ]
        for fig_name in expected_figs:
            fig_path = ASSETS_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Missing chart: {fig_name}")
            self.assertGreater(fig_path.stat().st_size, 1000, f"Chart file is too small: {fig_name}")


if __name__ == "__main__":
    unittest.main()
