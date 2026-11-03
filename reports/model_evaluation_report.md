# Phase 3: Machine Learning Model Development & Evaluation Report

**Project**: Lead Intelligence – Web-Based Lead Conversion Prediction Platform  
**Target Variable**: `Converted` (1 = Lead Enrolled, 0 = Lead Did Not Enroll)  
**Split Protocol**: Stratified 80:20 Split (Training N=7,392; Test N=1,848; Random State=42)  
**Validation Protocol**: 5-Fold Stratified Cross-Validation on Training Set  
**Model Persistence**: Best Pipeline saved at `models/trained_pipeline.joblib`  

---

## 1. Executive Summary & Model Selection

Four Scikit-Learn classification algorithms were trained, cross-validated, and tested:
1. **Dummy Classifier** (Baseline standard — predicts majority class `0`)
2. **Logistic Regression** (Linear baseline with L2 regularization)
3. **Decision Tree Classifier** (Interpretable non-linear tree)
4. **Random Forest Classifier** (Ensemble of bagged decision trees)

### Overall Performance Summary (Test Set: N=1,848)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Brier Score (Calibration) | 5-Fold CV Accuracy (Train) | 5-Fold CV ROC-AUC (Train) |
|---|---|---|---|---|---|---|---|---|
| **Dummy Classifier** | 61.47% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.3853 | 61.46% $\pm$ 0.00% | 0.5000 |
| **Logistic Regression** | 78.46% | 76.17% | 64.19% | 69.66% | **0.8505** | 0.1517 | 80.28% $\pm$ 0.73% | 0.8678 |
| **Decision Tree** | 78.95% | 73.51% | **70.93%** | **0.7219** | **0.8457** | 0.1494 | 80.88% $\pm$ 0.59% | 0.8604 |
| **Random Forest (Champion)** | **79.49%** | **77.70%** | 65.59% | **0.7113** | **0.8618** | **0.1479** | **81.22% $\pm$ 0.44%** | **0.8792** |

![Model Metrics Comparison](file:///c:/Web_Based_Merge/assets/models/model_metrics_comparison.png)

### Model Champion Decision:
**Random Forest Classifier** was selected as the champion production model because:
1. **Highest Discriminative Ability**: Achieved highest ROC-AUC (**0.8618** test / **0.8792** CV), indicating superior ability to rank higher-converting leads above lower-converting leads across all decision thresholds.
2. **Highest Precision (77.70%)**: Essential for sales team efficiency. Over 77 out of 100 leads flagged as high priority genuinely convert, minimizing wasted sales calls.
3. **Lowest Brier Score (0.1479)**: Demonstrates the most accurate and well-calibrated conversion probability scores.
4. **Cross-Validation Stability**: Lowest CV standard deviation ($\pm 0.44\%$) and minimal generalization gap ($81.22\%$ train CV vs $79.49\%$ test), proving zero overfitting.

---

## 2. Business Perspective: Metric Selection for Lead Prioritization

Why we do not rely purely on Accuracy:
- The dataset has a baseline non-conversion rate of 61.46%. A naive Dummy Classifier that rejects 100% of leads achieves a misleading 61.47% Accuracy while catching **0% of converting customers (Recall = 0%)**.
- In business lead management:
  - **Precision** measures efficiency: When sales reps call prioritized leads, how often do they reach a genuine buyer? High precision prevents sales burnout.
  - **Recall** measures coverage: What proportion of all possible buyers are identified? High recall prevents leaving money on the table.
  - **F1-Score & ROC-AUC** strike the optimal commercial trade-off between lead volume and sales team time expenditure.

---

## 3. Confusion Matrix Analysis (Test Set: N=1,848)

![Confusion Matrices](file:///c:/Web_Based_Merge/assets/models/confusion_matrices.png)

### Breakdown of Test Outcomes:

```
Actual Class Counts:
- Converted (1): 712 leads
- Not Converted (0): 1,136 leads
```

#### Random Forest Classifier:
- **True Negatives (TN)**: **1,002** (Correctly filtered out non-converters)
- **False Positives (FP)**: **134** (Non-converters mistakenly prioritized)
- **False Negatives (FN)**: **245** (Converters missed in standard queue)
- **True Positives (TP)**: **467** (Successfully identified converters)

#### Logistic Regression:
- **TN**: 993 | **FP**: 143 | **FN**: 255 | **TP**: 457

#### Decision Tree:
- **TN**: 954 | **FP**: 182 | **FN**: 207 | **TP**: 505 (Higher recall at cost of 48 more false alarms)

#### Dummy Classifier:
- **TN**: 1,136 | **FP**: 0 | **FN**: 712 | **TP**: 0 (Complete operational failure)

---

## 4. ROC Curves & Discrimination Capacity

![ROC Curves](file:///c:/Web_Based_Merge/assets/models/roc_curves.png)

The ROC curves illustrate True Positive Rate (Recall) vs False Positive Rate ($1 - \text{Specificity}$) across all operating probability thresholds:
- The Random Forest curve tightly hugs the top-left quadrant with an Area Under the Curve (AUC) of **0.862**.
- Logistic Regression achieves a strong **0.851** AUC, confirming that linear boundaries also capture substantial variance.
- Both models vastly outperform random chance ($AUC = 0.500$).

---

## 5. Model Interpretability & Feature Drivers

![Feature Importance & Coefficients](file:///c:/Web_Based_Merge/assets/models/feature_importance.png)

### 5.1 Random Forest Feature Importances (Gini Impurity Reduction)
1. **`Total Time Spent on Website`** (~42% relative importance): By far the single most predictive continuous indicator of visitor intent.
2. **`Lead Origin: Lead Add Form`**: Very high branch purity for conversion.
3. **`Occupation: Working Professional`**: Cleanest demographic indicator of high willingness and ability to pay.
4. **`Lead Source: Reference` & `Welingak Website`**: Powerful partner referral signals.
5. **`Occupation: Unknown`**: Strong negative indicator (form drop-off signal).
6. **`TotalVisits` & `Page Views Per Visit`**: Moderate secondary engagement signals.

### 5.2 Logistic Regression Coefficients (Log-Odds Interpretation)
- **Positive Drivers ($\beta > 0$)**:
  - `Occupation: Working Professional` ($\beta \approx +2.48$): Multiplies conversion odds by $e^{2.48} \approx 11.9\times$ compared to baseline.
  - `Lead Origin: Lead Add Form` ($\beta \approx +2.15$): Increases conversion odds by $8.6\times$.
  - `Lead Source: Welingak Website` ($\beta \approx +2.02$): Increases conversion odds by $7.5\times$.
  - `Total Time Spent on Website` ($\beta \approx +1.08$ per standard deviation): Substantially elevates conversion probability.
- **Negative Drivers ($\beta < 0$)**:
  - `Do Not Email: Yes` ($\beta \approx -1.35$): Significantly decreases odds of conversion.
  - `Occupation: Unknown` ($\beta \approx -0.95$): Incomplete form fields lower conversion likelihood by ~61%.

---

## 6. Probability Calibration & Reliability

![Calibration Curves](file:///c:/Web_Based_Merge/assets/models/calibration_curves.png)

In production lead intelligence, sales leaders need honest probabilities (e.g. "If a lead is scored 80%, roughly 8 out of 10 such leads must actually convert").

- The calibration curve shows mean predicted probability vs empirical conversion fraction in 10 uniform probability bins.
- **Random Forest & Logistic Regression** curves track the diagonal 45-degree ideal line closely:
  - Leads predicted in the 20% bucket convert at ~18-22%.
  - Leads predicted in the 80% bucket convert at ~78-83%.
  - Brier score for Random Forest is **0.1479** (vs 0.3853 for naive baseline), proving high calibration quality without requiring post-hoc isotonic distortion.

---

## 7. Delivery & Deployment Artifacts
- Champion Pipeline: [`models/trained_pipeline.joblib`](file:///c:/Web_Based_Merge/models/trained_pipeline.joblib)
- Multi-Model Artifact Store: [`models/all_models_and_metrics.joblib`](file:///c:/Web_Based_Merge/models/all_models_and_metrics.joblib)
- Inference API: [`src/predict.py`](file:///c:/Web_Based_Merge/src/predict.py)
