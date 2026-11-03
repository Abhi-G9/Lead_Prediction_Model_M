"""
Phase 3 Modeling & Evaluation Automated Tests
Validates model persistence, evaluation metrics, pipeline integrity,
and inference prediction outputs.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.predict import predict_lead, load_inference_pipeline
from src.evaluate_model import ASSETS_MODELS_DIR


class TestPhase3ModelingAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_path = PROJECT_ROOT / "models" / "trained_pipeline.joblib"
        cls.artifacts_path = PROJECT_ROOT / "models" / "all_models_and_metrics.joblib"
        cls.artifacts = joblib.load(cls.artifacts_path)

    def test_pipeline_files_exist(self):
        """Verify saved pipeline and artifact joblib files exist and are non-empty."""
        self.assertTrue(self.pipeline_path.exists())
        self.assertTrue(self.artifacts_path.exists())
        self.assertGreater(self.pipeline_path.stat().st_size, 1000)

    def test_all_four_models_trained(self):
        """Verify all 4 required algorithms are present in artifacts."""
        expected_models = {
            "Dummy Classifier",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
        }
        self.assertEqual(set(self.artifacts["results"].keys()), expected_models)

    def test_dummy_classifier_baseline(self):
        """Verify Dummy Classifier behaves as expected for majority baseline."""
        dummy_res = self.artifacts["results"]["Dummy Classifier"]
        self.assertAlmostEqual(dummy_res["test_roc_auc"], 0.5, places=3)
        self.assertEqual(dummy_res["test_precision"], 0.0)
        self.assertEqual(dummy_res["test_recall"], 0.0)

    def test_ml_models_beat_baseline(self):
        """Verify ML algorithms significantly beat Dummy baseline on ROC-AUC and F1."""
        dummy_acc = self.artifacts["results"]["Dummy Classifier"]["test_accuracy"]
        for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            res = self.artifacts["results"][name]
            self.assertGreater(res["test_accuracy"], dummy_acc, f"{name} failed to beat dummy accuracy")
            self.assertGreater(res["test_roc_auc"], 0.80, f"{name} ROC-AUC below 0.80")
            self.assertGreater(res["test_f1"], 0.65, f"{name} F1 below 0.65")
            self.assertLess(res["brier_score"], 0.20, f"{name} Brier score exceeds 0.20")

    def test_confusion_matrix_dimensions(self):
        """Verify confusion matrix for all models sums to 1,848 (20% of 9,240)."""
        for name, res in self.artifacts["results"].items():
            cm = np.array(res["confusion_matrix"])
            self.assertEqual(cm.shape, (2, 2), f"Invalid CM shape for {name}")
            self.assertEqual(cm.sum(), 1848, f"CM sum does not match test set size for {name}")

    def test_stratified_cv_metrics(self):
        """Verify 5-Fold Stratified Cross-Validation metrics exist and show stability."""
        for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            cv_summary = self.artifacts["results"][name]["cv_summary"]
            self.assertIn("cv_accuracy_mean", cv_summary)
            self.assertIn("cv_roc_auc_mean", cv_summary)
            # Ensure CV standard deviation is reasonably small (< 0.03) indicating stability
            self.assertLess(cv_summary["cv_accuracy_std"], 0.03)

    def test_single_record_inference(self):
        """Verify predict_lead() handles single dictionary input and returns required fields."""
        sample = {
            "TotalVisits": 3.0,
            "Total Time Spent on Website": 800,
            "Page Views Per Visit": 2.5,
            "Lead Origin": "Landing Page Submission",
            "Lead Source": "Google",
            "Do Not Email": "No",
            "Specialization": "Marketing Management",
            "What is your current occupation": "Working Professional",
            "A free copy of Mastering The Interview": "Yes",
        }
        res = predict_lead(sample)
        self.assertIn("predicted_class", res)
        self.assertIn(res["predicted_class"], [0, 1])
        self.assertIn("conversion_probability", res)
        self.assertGreaterEqual(res["conversion_probability"], 0.0)
        self.assertLessEqual(res["conversion_probability"], 1.0)
        self.assertIn("priority_tier", res)
        self.assertIn("disclaimer", res)

    def test_evaluation_charts_exist(self):
        """Verify all 5 model evaluation charts were generated and saved."""
        expected_figs = [
            "roc_curves.png",
            "confusion_matrices.png",
            "model_metrics_comparison.png",
            "feature_importance.png",
            "calibration_curves.png",
        ]
        for fig_name in expected_figs:
            fig_path = ASSETS_MODELS_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Missing evaluation chart: {fig_name}")
            self.assertGreater(fig_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
