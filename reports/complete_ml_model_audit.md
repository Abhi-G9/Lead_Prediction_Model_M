# COMPLETE MACHINE LEARNING MODEL AUDIT
## Project: Lead Intelligence – Lead Conversion Prediction Platform
**Target Audience**: Technical Review Board, College Project Evaluators & Job Interviewers  
**Dataset Source**: X Education Lead Scoring Dataset  
**Production Champion Model**: Scikit-Learn Random Forest Pipeline  

---

### SECTION 01: DATASET INFORMATION
- **Total Records (Rows)**: `9,240` leads
- **Total Raw Columns**: `37` features
- **Target Variable**: `Converted` (Binary: `1` = Converted/Enrolled, `0` = Not Converted)
- **Target Distribution**:
  - `0` (Did Not Convert): **5,679 records (61.46%)**
  - `1` (Converted): **3,561 records (38.54%)**
  - Target Missing/Null Values: **0 (0.0%)** — 100% clean, ground-truth labels.
- **Duplicate Records**:
  - Complete Duplicate Rows: **0**
  - Duplicate `Prospect ID`: **0** (9,240 unique UUIDs)
  - Duplicate `Lead Number`: **0** (9,240 unique sequential IDs)
- **Missing & Placeholder ("Select") Profiling**:
  - `How did you hear about X Education`: 78.46% unusable (5,043 "Select" + 2,207 nulls)
  - `Lead Profile`: 74.19% unusable (4,146 "Select" + 2,709 nulls)
  - `Lead Quality`: 51.59% nulls (post-contact sales score)
  - `City`: 39.71% unusable (2,249 "Select" + 1,420 nulls)
  - `Specialization`: 36.58% unusable (1,942 "Select" + 1,438 nulls)
  - `What is your current occupation`: 29.11% nulls
  - `TotalVisits` & `Page Views Per Visit`: 1.48% (137 nulls each)
  - `Total Time Spent on Website`: 0 nulls.

---

### SECTION 02: FEATURE SELECTION & LEAKAGE GOVERNANCE
Features were selected based on **Inference Time Availability** (available at the exact instant a lead submits an inquiry).

#### 1. Features Selected for ML Pipeline (9 Features):
- **3 Numerical**: `TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit`
- **6 Categorical**: `Lead Origin`, `Lead Source`, `What is your current occupation`, `Specialization`, `Do Not Email`, `A free copy of Mastering The Interview`

#### 2. Features Strictly Removed & Technical Reasons:
- **`Tags` (CRITICAL LEAKAGE)**: Contains `"Closed by Horizzon"` (99.4% converted) and `"Ringing"` (97.2% not converted). These are notes added by sales reps *after* calling. Using them produces artificial 98% accuracy that completely collapses in production.
- **`Lead Quality` (CRITICAL LEAKAGE)**: Subjective assessment assigned by sales reps during phone qualification; 51.6% missing.
- **`Last Activity` & `Last Notable Activity` (ACTIVITY LEAKAGE)**: Post-inquiry events (`"Converted to Lead"`, `"SMS Sent"`). Unknown when a fresh lead arrives.
- **Zero Variance Columns**: `Magazine`, `Receive More Updates About Our Courses`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque` (100% constant `"No"`).
- **Near-Zero Variance (>99.8% constant)**: `Do Not Call` (99.98% No), `Newspaper Article`, `Digital Advertisement`, `Search`.
- **Excessive Missingness / Placeholders**: `How did you hear about X Education` (78.5% unselected), `Lead Profile` (74.2% unselected).
- **Record Identifiers**: `Prospect ID`, `Lead Number` (UUIDs/counters removed to avoid arbitrary memorization).

---

### SECTION 03: DATA PREPROCESSING PIPELINE
Constructed using Scikit-Learn `ColumnTransformer` inside a unified `Pipeline`. Preprocessing is **fitted strictly on training data** to prevent test leakage.

1. **Numerical Pipeline**:
   - **Imputation**: `SimpleImputer(strategy='median')` (imputes 1.48% missing visits/page views).
   - **Outlier Mitigation**: 99th percentile capping (`TotalVisits` $\le 17.0$, `Page Views Per Visit` $\le 9.0$) without deleting valid rows.
   - **Scaling**: `StandardScaler()` (z-score normalization: $\mu=0, \sigma=1$) essential for gradient and distance calculations.
2. **Categorical Pipeline**:
   - **Imputation**: `SimpleImputer(strategy='constant', fill_value='Unknown')`.
   - **Standardization**: Casing normalized (`google` $\rightarrow$ `Google`), rare categories (<100 rows) collapsed into `'Other_Source'`.
   - **Placeholder Mapping**: `"Select"` mapped to `'Not Specified'`.
   - **Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. Unseen categories during inference are automatically converted to all-zero vectors without breaking.
3. **Total Feature Space**: 9 raw inputs $\rightarrow$ **44 encoded numerical dimensions**.

---

### SECTION 04: TRAIN-TEST SPLIT
- **Partition Ratio**: **80% Training, 20% Testing**
- **Splitting Strategy**: `StratifiedShuffleSplit` (`stratify=y`, `random_state=42`)
  - Ensures both sets have identical conversion ratios.
- **Training Records**: **7,392 leads**
  - Converted (`1`): 2,849 (38.54%)
  - Not Converted (`0`): 4,543 (61.46%)
- **Testing Records (Held-Out)**: **1,848 leads**
  - Converted (`1`): 712 (38.53%)
  - Not Converted (`0`): 1,136 (61.47%)
- **Leakage Check**: Disjoint index sets ($\text{Train} \cap \text{Test} = \emptyset$). `StandardScaler` fitted on exactly 7,392 samples.

---

### SECTION 05: ALGORITHMS IMPLEMENTED & SELECTION RATIONALE
1. **Dummy Classifier (`strategy='most_frequent'`)**:
   - **Role**: Essential baseline benchmark. Always predicts majority class (`0`). Proves that high accuracy alone is meaningless if recall is 0.
2. **Logistic Regression (`solver='lbfgs', C=0.5`)**:
   - **Role**: Linear parametric baseline. Highly interpretable through log-odds coefficients ($\beta$). Fast and memory-efficient.
3. **Decision Tree Classifier (`max_depth=5, min_samples_leaf=15`)**:
   - **Role**: Non-linear, rule-based baseline. Captures non-linear thresholds and interactions.
4. **Random Forest Classifier (`n_estimators=150, max_depth=7, min_samples_leaf=8`)**:
   - **Role (CHAMPION)**: Ensemble bagging of 150 regularized trees. Reduces variance, eliminates overfitting, models non-linear interactions, and delivers best probability calibration.

---

### SECTION 06: MODEL TRAINING & CROSS-VALIDATION
- **Cross-Validation**: **5-Fold Stratified K-Fold** executed strictly on the 7,392 training samples.
- **Hyperparameter Regularization**:
  - Tree depth constrained to 5–7 levels to prevent deep branch memorization.
  - `min_samples_leaf` set to 8–15 to prevent single-instance leaf splits.
  - L2 penalty on Logistic Regression with $C=0.5$.
- **Generalization Audit**:
  - Random Forest 5-Fold CV Accuracy: **81.22% $\pm$ 0.44%** | Test Accuracy: **79.49%** (Generalization Gap: **1.73%**)
  - Random Forest 5-Fold CV ROC-AUC: **0.8792** | Test ROC-AUC: **0.8618** (AUC Gap: **0.0174**)
  - Proves zero overfitting.

---

### SECTION 07: COMPREHENSIVE MODEL EVALUATION (HELD-OUT TEST SET: N=1,848)

| Evaluation Metric | Dummy Classifier | Logistic Regression | Decision Tree | Random Forest (Champion) |
|---|---|---|---|---|
| **Accuracy** | 61.47% | 78.46% | 78.95% | **79.49%** |
| **Precision** | 0.00% | 76.17% | 73.51% | **77.70%** |
| **Recall** | 0.00% | 64.19% | **70.93%** | 65.59% |
| **F1-Score** | 0.0000 | 0.6966 | **0.7219** | 0.7113 |
| **ROC-AUC** | 0.5000 | 0.8505 | 0.8457 | **0.8618** |
| **Brier Score (Calibration)** | 0.3853 | 0.1517 | 0.1494 | **0.1479** |

#### Why Random Forest Won:
- **Highest ROC-AUC (0.8618)**: Best capability to rank high-propensity leads ahead of low-propensity leads.
- **Highest Precision (77.70%)**: Crucial for sales ROI — when the model flags a lead as high priority, **78 out of 100 leads convert**, minimizing wasted sales rep calling hours.
- **Lowest Brier Score (0.1479)**: Lowest mean squared probability error, ensuring output probabilities are realistic.

---

### SECTION 08: DETAILED ERROR ANALYSIS & CONFUSION MATRIX

#### Confusion Matrix for Random Forest (Test Set: N=1,848):
- **True Negatives (TN)**: **1,002** (Correctly identified non-converters)
- **False Positives (FP)**: **134** (Non-converters incorrectly prioritized — Type I error)
- **False Negatives (FN)**: **245** (Converters missed in standard queue — Type II error)
- **True Positives (TP)**: **467** (Successfully converted leads correctly prioritized)

```
                       PREDICTED 0       PREDICTED 1
ACTUAL 0 (Lost)           1,002              134       (Total: 1,136)
ACTUAL 1 (Converted)        245              467       (Total: 712)
```

#### Detailed Classification Report:
- **Class 0 (Non-Converters)**: Precision = **80.35%**, Recall = **88.20%**, F1 = **0.8410**
- **Class 1 (Converters)**: Precision = **77.70%**, Recall = **65.59%**, F1 = **0.7113**
- **Macro Average**: F1 = **0.7762** | **Weighted Average**: F1 = **0.7910**

---

### SECTION 09: MODEL EXPLAINABILITY & PROBABILITY CALIBRATION

#### 1. Top Feature Drivers (Random Forest Gini Importance):
1. **`Total Time Spent on Website` (~42% importance)**: Single most dominant continuous intent driver.
2. **`Lead Origin: Lead Add Form`**: High branch purity (inbound direct forms).
3. **`Occupation: Working Professional`**: Strongest demographic indicator (purchasing power).
4. **`Lead Source: Reference` & `Welingak Website`**: Trust-based referral channels.
5. **`Occupation: Unknown`**: Strongest negative indicator (signals form drop-off / low intent).

#### 2. Logistic Regression Directional Odds ($\beta$ Coefficients):
- `Occupation: Working Professional` ($\beta = +2.48$): **Multiplies conversion odds by $11.9\times$**.
- `Lead Origin: Lead Add Form` ($\beta = +2.15$): Increases conversion odds by **$8.6\times$**.
- `Lead Source: Welingak Website` ($\beta = +2.02$): Increases conversion odds by **$7.5\times$**.
- `Do Not Email: Yes` ($\beta = -1.35$): Reduces conversion odds by **$74\%$**.
- `Occupation: Unknown` ($\beta = -0.95$): Reduces conversion odds by **$61\%$**.

#### 3. Probability Calibration (Reliability Diagram):
- In uniform 10-quantile bins, predicted probability closely tracks empirical conversion rate:
  - Bins predicted at 20% have ~19% actual conversion.
  - Bins predicted at 80% have ~81% actual conversion.
  - Champion Brier score is **0.1479** (vs 0.3853 for uncalibrated naive model).

---

### SECTION 10: PREDICTION VERIFICATION, TIERS & LIMITATIONS

#### 1. Operational Routing Tiers:
- **🟢 Tier 1: High Priority (Score $\ge 70\%$)**:
  - Example: Working Professional, Referral lead, 1200s on site $\rightarrow$ **Estimated Probability: ~94.7%**.
  - **SLA**: Sales outreach within **1 hour**.
- **🟡 Tier 2: Medium Priority ($40\% \le \text{Score} < 70\%$)**:
  - Example: Unemployed graduate, Google inbound, 400s on site $\rightarrow$ **Estimated Probability: ~52.1%**.
  - **SLA**: Standard follow-up within **24 hours**.
- **⚪ Tier 3: Low Priority (Score $< 40\%$)**:
  - Example: Unknown job, Olark Chat, 30s dwell time $\rightarrow$ **Estimated Probability: ~5.2%**.
  - **SLA**: Automated email drip campaign; zero manual phone bandwidth wasted.

#### 2. Real-World Limitations to State to Evaluators:
1. **Single-Point Scoring**: Captures intent at form submission; does not track multi-session return journeys across subsequent months.
2. **Geographic Concentration**: 95.8% of location data is from India (Mumbai/Maharashtra focus); expansion into international markets requires retraining.
3. **Correlation $\ne$ Causation**: High dwell time correlates with conversion, but forcing users to stay on a page will not artificially cause them to buy. Scores are prioritization estimates, not commercial closing guarantees.
