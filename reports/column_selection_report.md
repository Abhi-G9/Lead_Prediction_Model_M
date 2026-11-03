# Phase 1: Column Selection & Feature Governance Report

**Project**: Lead Intelligence – Web-Based Lead Conversion Prediction Platform  
**Dataset**: X Education Lead Scoring Dataset  
**Purpose**: Document the governance, technical justification, and inference-time availability for every column in the dataset.

---

## 1. Governance Principles

To ensure our Machine Learning models solve the real-world business objective (predicting conversion likelihood at the time a new lead is captured) rather than memorizing historical post-sales interactions, feature selection adheres to three strict principles:

1. **Inference Availability**: The feature must be legitimately known or capturable at the moment the lead is generated (e.g. from the web form submission or initial session analytics).
2. **Leakage Prevention**: No feature may contain information produced during or after subsequent sales representative touchpoints, qualification calls, or closing operations.
3. **Statistical Validity**: Features must exhibit sufficient informational variance (>0.5% variance) and maintain non-pathological missingness (<40% unrecoverable nulls).

---

## 2. Comprehensive Column Audit Table (All 37 Columns)

| # | Column Name | Decision | Primary Reason | Leakage Risk | Inference Availability |
|---|---|---|---|---|---|
| 1 | `Prospect ID` | **Excluded** | Arbitrary primary key (UUID); causes high-cardinality memorization. Retained solely for record lookup. | None | N/A |
| 2 | `Lead Number` | **Excluded** | Sequential system counter; non-predictive. | None | N/A |
| 3 | `Lead Origin` | **Kept** | Key acquisition channel indicator (`API`, `Landing Page Submission`, `Lead Add Form`, `Lead Import`). | None | Present at lead creation |
| 4 | `Lead Source` | **Kept** | Originating marketing medium (`Google`, `Direct Traffic`, `Olark Chat`, `Organic Search`, `Reference`, etc.). | None | Captured at session entry |
| 5 | `Do Not Email` | **Kept** | Direct contact preference explicitly declared by lead on submission form. | None | Form checkbox |
| 6 | `Do Not Call` | **Excluded** | Near-zero variance: 9,238 "No" vs 2 "Yes" (99.98% constant). | None | Form checkbox |
| 7 | `Converted` | **Target** | Ground truth label (1 = Converted, 0 = Not Converted). **Never an input feature.** | N/A | Target |
| 8 | `TotalVisits` | **Kept** | Numerical web engagement count. Subject to 99th percentile outlier clipping. | None | Web analytics session data |
| 9 | `Total Time Spent on Website` | **Kept** | Continuous numerical measure of visitor engagement (seconds). | None | Web analytics session data |
| 10 | `Page Views Per Visit` | **Kept** | Numerical measure of browsing depth. Subject to 99th percentile outlier capping. | None | Web analytics session data |
| 11 | `Last Activity` | **Excluded (Conditional)** | Activity leakage: contains post-contact events like `"Converted to Lead"` (428 records), `"Email Opened"`, and `"SMS Sent"`. | **High** | Occurs after lead creation |
| 12 | `Country` | **Excluded** | 26.6% missing; remaining records are 95.8% concentrated in `"India"` (near-zero geographic variance). | Low | Geo-IP lookup |
| 13 | `Specialization` | **Kept** | Educational/professional domain interest. Missing & "Select" combined into `"Not Specified"` category. | None | Dropdown on form |
| 14 | `How did you hear about X Education` | **Excluded** | 78.46% unusable (5,043 "Select" placeholders + 2,207 missing). | None | Dropdown on form |
| 15 | `What is your current occupation` | **Kept** | High business signal (`Unemployed`, `Working Professional`, `Student`, etc.). Missing grouped into `"Unknown"`. | None | Dropdown on form |
| 16 | `What matters most to you in choosing a course` | **Excluded** | 29.3% missing; non-null records are 99.95% single category (`"Better Career Prospects"`). Zero predictive variance. | None | Form field |
| 17 | `Search` | **Excluded** | Near-zero variance: 9,226 "No" vs 14 "Yes" (99.85% constant). | None | Marketing flag |
| 18 | `Magazine` | **Excluded** | Zero variance: 100% "No" (constant). | None | Marketing flag |
| 19 | `Newspaper Article` | **Excluded** | Near-zero variance: 9,238 "No" vs 2 "Yes" (99.98% constant). | None | Marketing flag |
| 20 | `X Education Forums` | **Excluded** | Near-zero variance: 9,239 "No" vs 1 "Yes" (99.99% constant). | None | Marketing flag |
| 21 | `Newspaper` | **Excluded** | Near-zero variance: 9,239 "No" vs 1 "Yes" (99.99% constant). | None | Marketing flag |
| 22 | `Digital Advertisement` | **Excluded** | Near-zero variance: 9,236 "No" vs 4 "Yes" (99.96% constant). | None | Marketing flag |
| 23 | `Through Recommendations` | **Excluded** | Near-zero variance: 9,233 "No" vs 7 "Yes" (99.92% constant). | None | Marketing flag |
| 24 | `Receive More Updates About Our Courses` | **Excluded** | Zero variance: 100% "No" (constant). | None | Form checkbox |
| 25 | `Tags` | **Excluded** | **Critical Target Leakage**. Populated by sales reps after calling leads (`"Closed by Horizzon"`, `"Ringing"`, `"Will revert..."`). | **Critical** | Post-contact sales notes |
| 26 | `Lead Quality` | **Excluded** | **Critical Target Leakage**. 51.6% missing; subjective rating given by sales rep during phone outreach. | **Critical** | Post-contact evaluation |
| 27 | `Update me on Supply Chain Content` | **Excluded** | Zero variance: 100% "No" (constant). | None | Form checkbox |
| 28 | `Get updates on DM Content` | **Excluded** | Zero variance: 100% "No" (constant). | None | Form checkbox |
| 29 | `Lead Profile` | **Excluded** | 74.19% unusable (4,146 "Select" + 2,709 missing). Derived post-inquiry. | Medium | Derived during qualification |
| 30 | `City` | **Excluded** | 39.71% unusable ("Select" + missing); heavily dominated by Mumbai. High overlap with Lead Origin. | Low | Dropdown on form |
| 31 | `Asymmetrique Activity Index` | **Excluded** | 45.65% missing; external agency scoring score not available for organic inbound leads. | Low | Third-party vendor |
| 32 | `Asymmetrique Profile Index` | **Excluded** | 45.65% missing; external scoring metric. | Low | Third-party vendor |
| 33 | `Asymmetrique Activity Score` | **Excluded** | 45.65% missing; external proprietary scoring model. | Low | Third-party vendor |
| 34 | `Asymmetrique Profile Score` | **Excluded** | 45.65% missing; external proprietary scoring model. | Low | Third-party vendor |
| 35 | `I agree to pay the amount through cheque` | **Excluded** | Zero variance: 100% "No" (constant). | None | Form checkbox |
| 36 | `A free copy of Mastering The Interview` | **Kept** | Form engagement flag (Yes/No); available at time of initial submission. | None | Form checkbox |
| 37 | `Last Notable Activity` | **Excluded (Conditional)** | Activity leakage: post-outreach logs (`"SMS Sent"`, `"Email Opened"`, `"Modified"`). | **High** | Follow-up tracking |

---

## 3. Summary of Selected Feature Set for Production Modeling

The final curated feature set consists of **8 high-integrity, leakage-free features** that are 100% accessible at the moment a lead enters the funnel:

### Numerical Features (3):
1. `TotalVisits` (Continuous / Discrete count of site visits)
2. `Total Time Spent on Website` (Continuous seconds on platform)
3. `Page Views Per Visit` (Continuous ratio of content consumption)

### Categorical Features (5):
4. `Lead Origin` (e.g., API, Landing Page Submission, Lead Add Form)
5. `Lead Source` (Cleaned & consolidated acquisition source)
6. `What is your current occupation` (e.g., Working Professional, Unemployed, Student)
7. `Specialization` (Domain interest area; missing/Select mapped to "Not Specified")
8. `Do Not Email` (Contact preference flag: Yes/No)

*(Note: `A free copy of Mastering The Interview` will be retained in data exploration and evaluated during pipeline feature selection).*

This curated feature specification strictly ensures zero target leakage, production readiness, and realistic conversion estimation.
