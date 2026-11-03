# Lead Intelligence — Web-Based Lead Conversion Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-43%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Portfolio-lightgrey.svg)]()

> A production-grade, end-to-end Machine Learning web application and data intelligence platform designed to estimate lead conversion probabilities, prioritize sales team workflows, and uncover commercial growth drivers.

---

## 1. Project Overview

In education and B2B services, acquiring customer inquiries is expensive, yet only a fraction of leads ever enroll. Without intelligent prioritization, sales representatives waste up to 70% of their outreach time calling unresponsive contacts while high-intent prospects go cold.

**Lead Intelligence** is a complete, leakage-free data science platform that analyzes historical customer engagement telemetry and predicts the conversion likelihood of incoming leads in real time. It routes each lead into actionable operational tiers (**High Priority**, **Medium Priority**, **Low Priority**) and equips sales leaders with empirical insights to maximize conversion efficiency.

---

## 2. Business Problem

Sales teams face four operational bottlenecks:
1. **Inbound Volume Overload**: Hundreds of inquiries arrive daily from disparate acquisition channels without clear indicators of purchase intent.
2. **Resource Misallocation**: Equal sales time is expended on low-propensity leads who only browse casually.
3. **Lack of Behavioral Transparency**: Teams do not know which demographic attributes or digital behaviors correlate with successful enrollment.
4. **Subjective Prioritization**: Sales outreach order relies on intuition rather than data-driven probability.

### Core Business Question:
> *"Which leads are more likely to convert, and which factors are associated with successful lead conversion?"*

---

## 3. Project Objectives

- **Develop a Leakage-Free ML Engine**: Train Scikit-Learn classification algorithms using exclusively features available at the moment of lead inquiry.
- **Empirical Probability Calibration**: Output honest, well-calibrated conversion probabilities ($0.0\%$ to $100.0\%$) rather than opaque hard thresholds.
- **Explainable Predictions**: Interpret global and local feature drivers using tree Gini importance and logistic log-odds coefficients.
- **Interactive Web Platform**: Deliver a multi-page Streamlit application for executive analytics, real-time lead scoring, technical model governance, and sales playbook execution.
- **Rigorous Software Verification**: Enforce 100% test coverage across data loading, transformation, modeling, edge-case robustness, and UI compilation.

---

## 4. Dataset Description

The application is built on the **X Education Lead Scoring Dataset** (`data/original/Leads.csv`), containing historical inquiry and conversion data for online training courses.

- **Total Inbound Leads**: 9,240 records
- **Total Attributes Audited**: 37 columns
- **Target Variable**: `Converted` (Binary: `1` = Enrolled, `0` = Did Not Enroll)
- **Class Balance**: 
  - `0` (Not Converted): **5,679 leads (61.46%)**
  - `1` (Converted): **3,561 leads (38.54%)**
- **Entity Uniqueness**: 9,240 unique `Prospect ID` UUIDs and `Lead Number` integers (zero duplicate records).

---

## 5. Dataset Limitations & Ethical Governance

1. **Academic Context**: The dataset represents historical online education inquiries from X Education. It is not proprietary customer data from any external consultancy.
2. **Estimation vs Guarantee**: Machine learning outputs are statistical probability estimates based on observed correlations. A predicted score is never presented as a guaranteed commercial sale.
3. **Temporal Invariance**: The data represents a fixed historical window. Changes in macroeconomic conditions, course pricing, or branding would require periodic model retraining.

---

## 6. Technology Stack

Strict adherence to lightweight, production-grade Python data science libraries:

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11+ / 3.14 | Core language runtime |
| **Web Application** | Streamlit | 1.35+ | Interactive multi-page dashboard UI |
| **Data Processing** | Pandas, NumPy | 2.2+, 1.26+ | Data audit, cleaning, vector operations |
| **Machine Learning** | Scikit-Learn | 1.4+ | Preprocessing pipelines, classification, CV |
| **Data Visualization** | Matplotlib, Seaborn | 3.8+, 0.13+ | Publication-quality statistical charts |
| **Model Persistence** | Joblib | 1.3+ | Pipeline serialization & deserialization |
| **Testing** | Unittest | Standard Lib | 43 automated verification tests |

*Zero unapproved technologies (no TensorFlow, no PyTorch, no XGBoost, no external paid APIs, no synthetic data generators).*

---

## 7. Machine Learning Workflow

```
[Raw Data: Leads.csv] (data/original/)
         │
         ▼
[Phase 1: Comprehensive Data Audit & Leakage Detection]
         │ (Reports: data_audit_report.md, column_selection_report.md)
         ▼
[Phase 2: Reproducible Data Cleaning & EDA]
         │ (data/processed/cleaned_leads.csv, 7 EDA charts)
         ▼
[Phase 3: Pipeline Feature Engineering & Model Training]
         │ ├── ColumnTransformer (StandardScaler + OneHotEncoder)
         │ ├── Stratified 80:20 Split (Train: 7,392 | Test: 1,848)
         │ └── 5-Fold Stratified Cross-Validation on Train
         ▼
[Phase 4: Independent QA, Recalculation & Stress Testing]
         │ (14 QA tests, Zero leakage verification, 38 total tests)
         ▼
[Phase 5: Multi-Page Streamlit Web Application]
         │ (Executive KPI Dashboard, Analytics, Real-Time Scoring, Playbook)
         ▼
[Phase 6: Final Integration, Packaging & Delivery]
```

---

## 8. Feature Selection & Leakage Prevention

### Excluded Due to Target & Activity Leakage:
- **`Tags`**: Post-sales tags like `"Closed by Horizzon"` (99.4% conversion) and `"Ringing"` (97.2% non-conversion) are recorded by sales reps *during/after* follow-up calls. Including them causes catastrophic target leakage (>95% fake accuracy).
- **`Lead Quality`**: Subjective rating assigned by reps during qualification.
- **`Last Activity` & `Last Notable Activity`**: Logged interactions after lead arrival (e.g., `"Converted to Lead"`, `"SMS Sent"`).

### Excluded Due to Zero / Near-Zero Variance:
- `Magazine`, `Receive More Updates...`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque` (100% constant `"No"`).
- `Do Not Call`, `Newspaper Article`, `Digital Advertisement` (>99.9% constant `"No"`).

### Selected Curated Feature Set (100% Known at Lead Arrival):
1. **`TotalVisits`** (Numerical — Site visit count)
2. **`Total Time Spent on Website`** (Numerical — Active engagement in seconds)
3. **`Page Views Per Visit`** (Numerical — Browsing depth ratio)
4. **`Lead Origin`** (Categorical — Landing Page, API, Lead Add Form, Lead Import)
5. **`Lead Source`** (Categorical — Google, Direct Traffic, Olark Chat, Reference, etc.)
6. **`What is your current occupation`** (Categorical — Working Professional, Unemployed, Student, etc.)
7. **`Specialization`** (Categorical — Domain interest; "Select"/nulls mapped to "Not Specified")
8. **`Do Not Email`** (Categorical — Contact preference flag)
9. **`A free copy of Mastering The Interview`** (Categorical — Engagement checkbox)

---

## 9. Data Cleaning Process

- **Pristine Raw Preservation**: `data/original/Leads.csv` is never overwritten. All cleaning executes on a distinct working copy.
- **Zero Record Loss**: Retained **100% of customer records (all 9,240 rows)**. No valid customer leads were discarded.
- **Placeholder Resolution**: Form dropdowns with default `"Select"` strings were properly recognized as unselected and mapped to `'Not Specified'`.
- **Casing & Low-Frequency Consolidation**: Standardized casing (`google` $\rightarrow$ `Google`) and grouped rare sources (<100 occurrences) into `'Other_Source'`.
- **Robust Outlier Capping**: Imputed missing values with medians and clipped upper extreme outliers at the 99th percentile (`TotalVisits` $\le 17.0$, `Page Views Per Visit` $\le 9.0$) without deleting valid rows.

---

## 10. Model Comparison & Benchmark Results

Evaluated on held-out unseen test split ($N=1,848$, 20% stratified test set):

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Brier Score | 5-Fold Train CV Accuracy |
|---|---|---|---|---|---|---|---|
| **Dummy Classifier (Baseline)** | 61.47% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.3853 | 61.46% $\pm$ 0.00% |
| **Logistic Regression** | 78.46% | 76.17% | 64.19% | 69.66% | 0.8505 | 0.1517 | 80.28% $\pm$ 0.73% |
| **Decision Tree** | 78.95% | 73.51% | **70.93%** | **0.7219** | 0.8457 | 0.1494 | 80.88% $\pm$ 0.59% |
| **Random Forest (Champion)** | **79.49%** | **77.70%** | 65.59% | **0.7113** | **0.8618** | **0.1479** | **81.22% $\pm$ 0.44%** |

### Why Random Forest is the Production Champion:
- **Highest ROC-AUC (0.8618)**: Maximizes ranking discrimination across every operational cutoff.
- **Highest Precision (77.70%)**: Over 77 out of 100 prioritized leads convert, minimizing wasted rep call time.
- **Superior Calibration (Brier: 0.1479)**: Raw probability scores accurately reflect ground-truth conversion frequencies.
- **Zero Overfitting**: Minimal 1.73% generalization gap between training CV ($81.22\%$) and test ($79.49\%$).

---

## 11. Key Evaluation Metrics Explained

- **Accuracy**: $\frac{TP+TN}{Total}$. Baseline reference, but deceptive on imbalanced datasets.
- **Precision**: $\frac{TP}{TP+FP}$. Measures sales efficiency: when a lead is flagged high priority, does it convert?
- **Recall**: $\frac{TP}{TP+FN}$. Measures coverage: what percentage of all genuine buyers were captured?
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the True Positive Rate vs False Positive Rate curve across all thresholds.
- **Brier Score**: Mean squared difference between predicted probability and actual binary outcome ($0$ or $1$). Closer to 0 indicates superior probability calibration.

---

## 12. Application Structure & Pages

1. **Executive Overview (`app.py`)**: Dynamic KPI cards, problem definition, and governance summary.
2. **Lead Analytics (`pages/1_Lead_Analytics.py`)**: Multi-slicer interactive dashboard with dynamic conversion deltas.
3. **Real-Time Prediction (`pages/2_Lead_Prediction.py`)**: Form with pre-built demo scenarios, instant probability scoring, priority badges, and compliance notices.
4. **Model Performance (`pages/3_Model_Performance.py`)**: Benchmark comparison table, ROC curves, confusion matrices, feature importance, and calibration curves.
5. **Business Insights (`pages/4_Business_Insights.py`)**: Empirical conversion findings and the 3-Tier Sales Routing Playbook.

---

## 13. Installation Instructions

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git (optional)

### Setup
```bash
# 1. Clone or navigate to the project directory
cd c:\Web_Based_Merge

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 14. How to Run the Project

### Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Retrain Models or Re-run Pipelines (Optional)
```bash
# Run data audit
python src/data_audit.py

# Run cleaning pipeline
python src/data_cleaning.py

# Run exploratory analysis & regenerate EDA charts
python src/eda.py

# Train models, cross-validate & persist pipeline
python src/train_model.py

# Regenerate model evaluation figures
python src/evaluate_model.py
```

---

## 15. Testing Process & Test Suite

The project includes **43 automated unit and validation tests** across 5 test suites:

```bash
# Execute the complete automated test suite
python -m unittest discover tests
```

### Breakdown of Test Suites:
1. [`tests/test_phase1_audit.py`](file:///c:/Web_Based_Merge/tests/test_phase1_audit.py) (5 tests): Raw data existence, row/column counts, target distribution, ID uniqueness.
2. [`tests/test_phase2_cleaning.py`](file:///c:/Web_Based_Merge/tests/test_phase2_cleaning.py) (11 tests): 9,240 rows preserved, zero nulls, outlier clipping ($\le 17$ visits), casing normalization, 7 EDA charts.
3. [`tests/test_phase3_modeling.py`](file:///c:/Web_Based_Merge/tests/test_phase3_modeling.py) (8 tests): Multi-model persistence, baseline comparison, ROC-AUC $> 0.84$, confusion matrix dimensions.
4. [`tests/test_phase4_qa_validation.py`](file:///c:/Web_Based_Merge/tests/test_phase4_qa_validation.py) (14 tests): Train-test disjointness, zero leakage, independent recalculation, probability calibration, edge cases (0 visits, novel categories, empty payload).
5. [`tests/test_phase5_app.py`](file:///c:/Web_Based_Merge/tests/test_phase5_app.py) (5 tests): Syntax and compilation tests for `app.py` and all 4 pages.

**All 43 tests pass with 0 errors.**

---

## 16. Limitations & Future Roadmap

### Current Limitations:
- Single-point lead scoring: Telemetry is scored at session/form capture; does not model multi-session return journeys over months.
- Geolocation concentration: 95.8% of location data is within India; regional models may be needed for global scaling.

### Future Roadmap:
- **Real-Time Webhook Integration**: Connect directly to CRM platforms (HubSpot, Salesforce) via REST API.
- **Dynamic Cost-Sensitive Thresholding**: Allow sales managers to adjust probability cutoffs based on daily rep call capacity.
- **SHAP Local Explainability**: Integrate Shapley additive explanations for per-lead explanation waterfalls.

---

## 17. Presentation & Interview Quick-Reference (For BCA Students)

| Question | Recommended Answer |
|---|---|
| *"Why didn't you use Tags or Lead Quality?"* | "Because they are assigned by sales agents *after* outreach calls. Using them causes target leakage, yielding artificial 95% accuracy that collapses when predicting genuinely new leads." |
| *"Why Random Forest over Logistic Regression?"* | "Random Forest achieved higher ROC-AUC (0.862 vs 0.851), higher precision (77.7% vs 76.2%), and superior probability calibration without overfitting (CV accuracy within 1.7% of test)." |
| *"What is the strongest conversion signal?"* | "Total Time Spent on Website (+123.5% longer for converting leads) and Working Professional occupation (91.6% conversion rate)." |
| *"How are missing values handled?"* | "Numerical features use median imputation; unselected dropdowns ('Select') are mapped to an explicit 'Not Specified' category; no customer rows were deleted." |

---

*Lead Intelligence Platform • Senior Machine Learning & Software Engineering Project • Created November 2026*
