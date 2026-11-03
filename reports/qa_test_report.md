# Phase 4: QA & Independent Machine Learning Test Report

**Project**: Lead Intelligence – Web-Based Lead Conversion Prediction Platform  
**Test Suite**: `tests/test_phase4_qa_validation.py`  
**Test Harness**: Python `unittest`  
**Auditor**: Independent QA & ML Validation Engineer  
**Date**: November 2026  

---

## 1. Executive Summary

As required by Phase 4, an exhaustive QA and stress-testing campaign was executed on the data pipeline, preprocessing transformers, trained ML pipelines, and prediction APIs. 

- **Total Test Cases Executed in Phase 4**: 14
- **Initial Passes**: 12
- **Initial Failures/Errors**: 2
- **Fixes Applied & Verified**: 2
- **Final Test Status**: **14 PASSED, 0 FAILED (100% Pass Rate)**
- **Cumulative Project Test Suite**: **38 tests passing across all 4 phases**

---

## 2. Detailed Test Execution Log

| # | Test Name | Expected Result | Actual Result | Initial Status | Fix Applied | Final Status |
|---|---|---|---|---|---|---|
| 1 | `test_train_test_split_disjoint` | Train (7,392) and Test (1,848) index sets have zero intersection. | Overlap = 0; exactly 7,392 train and 1,848 test records. | **PASS** | None required. | **PASS** |
| 2 | `test_no_id_columns_in_model_features` | `Prospect ID` and `Lead Number` excluded from model inputs. | Neither column present in feature list. | **PASS** | None required. | **PASS** |
| 3 | `test_no_target_leakage_columns_in_model_features` | `Tags`, `Lead Quality`, `Last Activity`, etc., absent from model features. | Strictly 0 leakage columns present in pipeline feature space. | **PASS** | None required. | **PASS** |
| 4 | `test_preprocessor_does_not_leak_test_data` | Preprocessing scaler `n_samples_seen_` equals exactly 7,392 (training size only). | Initial run threw `IndexError: invalid index to scalar variable` due to `[0]` index on scalar integer. | **ERROR** | Updated type handling to read scalar integer directly (`int(num_scaler.n_samples_seen_) == 7392`). | **PASS** |
| 5 | `test_recalculate_metrics_from_raw_arrays` | Independently recalculated Accuracy, Precision, Recall, F1, and AUC match stored metrics. | Exact numerical match ($< 10^{-4}$ tolerance) across all models. | **PASS** | None required. | **PASS** |
| 6 | `test_manual_confusion_matrix_recalculation` | Manual boolean masking $(y=0, \hat{y}=0)$, etc., matches stored confusion matrices. | TP=467, TN=1002, FP=134, FN=245 match stored matrix exactly. | **PASS** | None required. | **PASS** |
| 7 | `test_generalization_gap_and_overfitting` | Train CV Accuracy vs Test Accuracy gap is $< 5\%$. | Random Forest: 81.22% CV vs 79.49% test (gap = 1.73%). Logistic Regression: 80.28% CV vs 78.46% test (gap = 1.82%). | **PASS** | None required. | **PASS** |
| 8 | `test_probability_range_and_calibration` | Probability outputs sum to 1.0 across classes; $P(y=1) \in [0.0, 1.0]$ with non-degenerate std ($> 0.10$). | Row sums = 1.00000; std = 0.287; range [0.009, 0.998]. | **PASS** | None required. | **PASS** |
| 9 | `test_edge_case_zero_activity` | Lead with 0 visits, 0 seconds on site produces low conversion probability ($< 0.45$). | Successfully predicted 18.2% conversion probability (Low Priority). | **PASS** | None required. | **PASS** |
| 10 | `test_edge_case_extreme_numerical_inputs` | Extreme values (500 visits, 8,000s on site) do not crash pipeline or produce NaN/overflow. | Handled gracefully without crash; bounded probability = 98.4%. | **PASS** | None required. | **PASS** |
| 11 | `test_edge_case_unseen_unknown_categories` | Novel categories (`'TikTok_Ad'`, `'Quantum_Physics'`) handled smoothly without exception. | Handled via `OneHotEncoder(handle_unknown='ignore')`; valid probability returned. | **PASS** | None required. | **PASS** |
| 12 | `test_edge_case_completely_empty_input` | Empty dictionary `{}` input uses default imputation and returns valid prediction payload. | Missing values imputed with medians/constants; valid prediction returned. | **PASS** | None required. | **PASS** |
| 13 | `test_batch_prediction_consistency` | Calling `predict_lead()` on a DataFrame produces identical probabilities to individual single-record calls. | Exact match ($< 10^{-5}$) across multiple records. | **PASS** | None required. | **PASS** |
| 14 | `test_model_idempotence_and_reloading` | Reloading `trained_pipeline.joblib` from disk generates identical predictions to in-memory instance. | Initial run threw `AssertionError: 0.5544279678412368 != 0.5544279678412369` ($10^{-16}$ float drift). | **FAIL** | Replaced `assertEqual` with `assertAlmostEqual(places=7)` for floating point safety. | **PASS** |

---

## 3. Errors Detected and Fixes Applied

### Defect 1: Scaler Sample Attribute Type Handling
- **Location**: `tests/test_phase4_qa_validation.py::test_preprocessor_does_not_leak_test_data`
- **Issue**: Modern Scikit-Learn versions store `n_samples_seen_` on `StandardScaler` as a scalar `int` rather than a numpy 1D array when fitted on a 2D matrix. Indexing `num_scaler.n_samples_seen_[0]` triggered an `IndexError`.
- **Fix**: Wrapped the attribute in an integer check:
  ```python
  n_samples = int(num_scaler.n_samples_seen_) if hasattr(num_scaler.n_samples_seen_, "__int__") else int(num_scaler.n_samples_seen_[0])
  self.assertEqual(n_samples, 7392)
  ```
- **Verification**: Re-test passed, confirming the scaler processed exactly the 7,392 training rows.

### Defect 2: Floating-Point Strict Equality on Model Reload
- **Location**: `tests/test_phase4_qa_validation.py::test_model_idempotence_and_reloading`
- **Issue**: Python `self.assertEqual(pred_before, pred_after)` failed due to a sub-atomic IEEE-754 precision difference ($0.5544279678412368$ vs $0.5544279678412369$, $\Delta = 1 \times 10^{-16}$) after pickle de-serialization.
- **Fix**: Replaced exact binary equality with standard numerical assertion:
  ```python
  self.assertAlmostEqual(pred_before, pred_after, places=7)
  ```
- **Verification**: Re-test passed, confirming complete idempotence down to 7 decimal digits.

---

## 4. Overall Conclusion
The QA verification demonstrates that the machine learning system is mathematically sound, robust against edge-case anomalies, and strictly free of target leakage and data snooping.
