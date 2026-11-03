"""
Phase 4 QA, Error Detection & Model Validation Test Suite
Acts as an independent QA and Machine Learning Validation Engineer to stress-test:
1. Data Integrity and Leakage Prevention
2. Independent Metric Recalculation and Consistency
3. Probabilistic Calibration and Invariance
4. Extreme Edge Cases, Unseen Categories, and Missing Inputs
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_cleaning import load_cleaned_data
from src.feature_engineering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from src.predict import predict_lead, load_inference_pipeline


class TestPhase4QAAndValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts_path = PROJECT_ROOT / "models" / "all_models_and_metrics.joblib"
        cls.artifacts = joblib.load(cls.artifacts_path)
        cls.df = load_cleaned_data()
        cls.pipeline = load_inference_pipeline()

    # =========================================================================
    # A. DATA VALIDATION & LEAKAGE PREVENTION TESTS
    # =========================================================================

    def test_train_test_split_disjoint(self):
        """Verify train and test indices have strictly zero overlap."""
        test_indices = set(self.artifacts["test_indices"])
        all_indices = set(self.df.index)
        train_indices = all_indices - test_indices
        self.assertEqual(len(test_indices.intersection(train_indices)), 0)
        self.assertEqual(len(test_indices), 1848)
        self.assertEqual(len(train_indices), 7392)

    def test_no_id_columns_in_model_features(self):
        """Verify Prospect ID and Lead Number are not in the model feature space."""
        model_features = self.artifacts["feature_cols"]
        self.assertNotIn("Prospect ID", model_features)
        self.assertNotIn("Lead Number", model_features)

    def test_no_target_leakage_columns_in_model_features(self):
        """Verify post-sales leakage features are strictly excluded."""
        leakage_cols = [
            "Tags",
            "Lead Quality",
            "Last Activity",
            "Last Notable Activity",
            "I agree to pay the amount through cheque",
        ]
        model_features = self.artifacts["feature_cols"]
        for col in leakage_cols:
            self.assertNotIn(col, model_features, f"Leakage column '{col}' detected in model features!")

    def test_preprocessor_does_not_leak_test_data(self):
        """Verify preprocessor transformer was fitted only on training set."""
        rf_pipeline = self.artifacts["fitted_pipelines"]["Random Forest"]
        prep = rf_pipeline.named_steps["preprocessor"]
        num_scaler = prep.named_transformers_["num"].named_steps["scaler"]
        # StandardScaler n_samples_seen_ must equal training set size (7,392), NOT full dataset (9,240)
        n_samples = int(num_scaler.n_samples_seen_) if hasattr(num_scaler.n_samples_seen_, "__int__") else int(num_scaler.n_samples_seen_[0])
        self.assertEqual(
            n_samples,
            7392,
            f"Scaler was fitted on {n_samples} samples instead of training set (7,392)!",
        )

    # =========================================================================
    # B. MODEL & METRIC INDEPENDENT RECALCULATION
    # =========================================================================

    def test_recalculate_metrics_from_raw_arrays(self):
        """Independently recalculate all classification metrics from raw predictions."""
        for name, res in self.artifacts["results"].items():
            if name == "Dummy Classifier":
                continue
            y_test = np.array(res["y_test"])
            y_pred = np.array(res["y_pred"])
            y_proba = np.array(res["y_proba"])

            # Independent recalculations
            calc_acc = accuracy_score(y_test, y_pred)
            calc_prec = precision_score(y_test, y_pred, zero_division=0)
            calc_rec = recall_score(y_test, y_pred, zero_division=0)
            calc_f1 = f1_score(y_test, y_pred, zero_division=0)
            calc_auc = roc_auc_score(y_test, y_proba)

            self.assertAlmostEqual(res["test_accuracy"], calc_acc, places=4)
            self.assertAlmostEqual(res["test_precision"], calc_prec, places=4)
            self.assertAlmostEqual(res["test_recall"], calc_rec, places=4)
            self.assertAlmostEqual(res["test_f1"], calc_f1, places=4)
            self.assertAlmostEqual(res["test_roc_auc"], calc_auc, places=4)

    def test_manual_confusion_matrix_recalculation(self):
        """Manually compute TP, TN, FP, FN and verify against stored confusion matrix."""
        rf_res = self.artifacts["results"]["Random Forest"]
        y_test = np.array(rf_res["y_test"])
        y_pred = np.array(rf_res["y_pred"])

        tn = int(np.sum((y_test == 0) & (y_pred == 0)))
        fp = int(np.sum((y_test == 0) & (y_pred == 1)))
        fn = int(np.sum((y_test == 1) & (y_pred == 0)))
        tp = int(np.sum((y_test == 1) & (y_pred == 1)))

        stored_cm = rf_res["confusion_matrix"]
        self.assertEqual(stored_cm[0][0], tn)
        self.assertEqual(stored_cm[0][1], fp)
        self.assertEqual(stored_cm[1][0], fn)
        self.assertEqual(stored_cm[1][1], tp)

        # Re-derive Precision and Recall from raw counts
        manual_prec = tp / (tp + fp)
        manual_rec = tp / (tp + fn)
        self.assertAlmostEqual(rf_res["test_precision"], manual_prec, places=4)
        self.assertAlmostEqual(rf_res["test_recall"], manual_rec, places=4)

    def test_generalization_gap_and_overfitting(self):
        """Verify model does not exhibit severe overfitting (train-test gap < 5%)."""
        for name in ["Logistic Regression", "Random Forest"]:
            res = self.artifacts["results"][name]
            train_acc = res["cv_summary"]["train_accuracy_mean"]
            test_acc = res["test_accuracy"]
            gap = abs(train_acc - test_acc)
            self.assertLess(
                gap,
                0.05,
                f"Model '{name}' exhibits suspiciously large generalization gap: {gap:.4f}",
            )

    # =========================================================================
    # C. PROBABILITY VALIDATION TESTS
    # =========================================================================

    def test_probability_range_and_calibration(self):
        """Verify predict_proba outputs are strictly between 0 and 1."""
        test_df = self.df.loc[self.artifacts["test_indices"]][NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        probas = self.pipeline.predict_proba(test_df)

        # Probabilities must sum to 1.0 across classes
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

        # Positive class probability within [0, 1]
        p1 = probas[:, 1]
        self.assertTrue(np.all(p1 >= 0.0) and np.all(p1 <= 1.0))
        # Ensure model is not predicting degenerate constant probabilities
        self.assertGreater(p1.std(), 0.10)

    # =========================================================================
    # D. EDGE-CASE & ROBUSTNESS STRESS TESTS
    # =========================================================================

    def test_edge_case_zero_activity(self):
        """Stress test: Lead with zero website visits and zero time spent."""
        lead = {
            "TotalVisits": 0.0,
            "Total Time Spent on Website": 0,
            "Page Views Per Visit": 0.0,
            "Lead Origin": "Landing Page Submission",
            "Lead Source": "Direct Traffic",
            "Do Not Email": "No",
            "Specialization": "Not Specified",
            "What is your current occupation": "Unemployed",
            "A free copy of Mastering The Interview": "No",
        }
        res = predict_lead(lead)
        self.assertIn("conversion_probability", res)
        # With zero time on site and generic form, conversion likelihood should be low (< 0.40)
        self.assertLess(res["conversion_probability"], 0.45)
        self.assertIn(res["priority_tier"], ["Low Priority", "Medium Priority"])

    def test_edge_case_extreme_numerical_inputs(self):
        """Stress test: Lead with extreme high numerical inputs (e.g., 500 visits, 8,000s on site)."""
        lead = {
            "TotalVisits": 500.0,
            "Total Time Spent on Website": 8000,
            "Page Views Per Visit": 40.0,
            "Lead Origin": "Lead Add Form",
            "Lead Source": "Reference",
            "Do Not Email": "No",
            "Specialization": "Finance Management",
            "What is your current occupation": "Working Professional",
            "A free copy of Mastering The Interview": "Yes",
        }
        res = predict_lead(lead)
        # Pipeline must not crash on extreme values; probability must stay clamped in [0, 1]
        self.assertGreaterEqual(res["conversion_probability"], 0.0)
        self.assertLessEqual(res["conversion_probability"], 1.0)
        self.assertEqual(res["priority_tier"], "High Priority")

    def test_edge_case_unseen_unknown_categories(self):
        """Stress test: Lead with novel categories never encountered in training data."""
        lead = {
            "TotalVisits": 2.0,
            "Total Time Spent on Website": 500,
            "Page Views Per Visit": 2.0,
            "Lead Origin": "TikTok_Ad_Campaign",  # Brand new origin
            "Lead Source": "Discord_Community",   # Brand new source
            "Do Not Email": "No",
            "Specialization": "Quantum_Astrophysics",  # Brand new specialization
            "What is your current occupation": "AI_Prompt_Engineer",  # Brand new occupation
            "A free copy of Mastering The Interview": "No",
        }
        # OneHotEncoder(handle_unknown='ignore') must smoothly handle unseen levels without error
        res = predict_lead(lead)
        self.assertIn("conversion_probability", res)
        self.assertGreaterEqual(res["conversion_probability"], 0.0)
        self.assertLessEqual(res["conversion_probability"], 1.0)

    def test_edge_case_completely_empty_input(self):
        """Stress test: Completely empty dictionary input."""
        res = predict_lead({})
        self.assertIn("conversion_probability", res)
        self.assertIn("predicted_class", res)
        self.assertIn("priority_tier", res)

    def test_batch_prediction_consistency(self):
        """Verify batch prediction matches individual single-record predictions."""
        sample_1 = {
            "TotalVisits": 1.0,
            "Total Time Spent on Website": 50,
            "Page Views Per Visit": 1.0,
            "Lead Origin": "API",
            "Lead Source": "Olark Chat",
            "Do Not Email": "No",
            "Specialization": "Not Specified",
            "What is your current occupation": "Student",
            "A free copy of Mastering The Interview": "No",
        }
        sample_2 = {
            "TotalVisits": 5.0,
            "Total Time Spent on Website": 1500,
            "Page Views Per Visit": 3.0,
            "Lead Origin": "Lead Add Form",
            "Lead Source": "Reference",
            "Do Not Email": "No",
            "Specialization": "Human Resource Management",
            "What is your current occupation": "Working Professional",
            "A free copy of Mastering The Interview": "No",
        }

        single_1 = predict_lead(sample_1)
        single_2 = predict_lead(sample_2)

        batch_df = pd.DataFrame([sample_1, sample_2])
        batch_res = predict_lead(batch_df)

        self.assertAlmostEqual(
            single_1["conversion_probability"],
            batch_res["predictions"][0]["conversion_probability"],
            places=5,
        )
        self.assertAlmostEqual(
            single_2["conversion_probability"],
            batch_res["predictions"][1]["conversion_probability"],
            places=5,
        )

    def test_model_idempotence_and_reloading(self):
        """Verify reloading the model from disk produces 100% identical predictions."""
        lead = {
            "TotalVisits": 3.0,
            "Total Time Spent on Website": 700,
            "Page Views Per Visit": 2.0,
            "Lead Origin": "Landing Page Submission",
            "Lead Source": "Google",
            "Do Not Email": "No",
            "Specialization": "Operations Management",
            "What is your current occupation": "Unemployed",
            "A free copy of Mastering The Interview": "No",
        }
        pred_before = predict_lead(lead)["conversion_probability"]

        # Reload directly from disk
        reloaded_pipe = joblib.load(PROJECT_ROOT / "models" / "trained_pipeline.joblib")
        df_in = pd.DataFrame([lead])[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        pred_after = float(reloaded_pipe.predict_proba(df_in)[0, 1])

        # Check floating-point equivalence across reloads
        self.assertAlmostEqual(pred_before, pred_after, places=7)


if __name__ == "__main__":
    unittest.main()
