# Phase 1: Comprehensive Data Audit Report

**Project**: Lead Intelligence – Web-Based Lead Conversion Prediction Platform  
**Dataset Source**: X Education Lead Scoring Dataset (`data/original/Leads.csv`)  
**Audit Date**: November 2026  
**Auditor**: Lead Intelligence Engineering & QA Team  

---

## 1. Executive Summary

A comprehensive data audit was conducted on the raw X Education Leads dataset. The dataset represents historical lead acquisition and sales engagement records for online education courses. 

- **Total Records (Rows)**: 9,240
- **Total Attributes (Columns)**: 37
- **Target Variable**: `Converted` (Binary: 0 or 1)
- **Target Distribution**:
  - `0` (Not Converted): 5,679 leads (**61.46%**)
  - `1` (Converted): 3,561 leads (**38.54%**)
  - Missing or Invalid Target Values: **0 (0.0%)** — The target is 100% clean and intact.
- **Duplicate Records**:
  - Full Duplicate Rows: **0**
  - Duplicate `Prospect ID`: **0** (9,240 unique identifiers)
  - Duplicate `Lead Number`: **0** (9,240 unique sequential IDs)

---

## 2. Dataset Schema and Data Types

The raw CSV consists of 37 columns across integer, floating-point, and string data types.

| # | Column Name | Data Type | Null Count | Null % | 'Select' Count | Total Missing/Select % |
|---|---|---|---|---|---|---|
| 1 | Prospect ID | string | 0 | 0.00% | 0 | 0.00% |
| 2 | Lead Number | int64 | 0 | 0.00% | 0 | 0.00% |
| 3 | Lead Origin | string | 0 | 0.00% | 0 | 0.00% |
| 4 | Lead Source | string | 36 | 0.39% | 0 | 0.39% |
| 5 | Do Not Email | string | 0 | 0.00% | 0 | 0.00% |
| 6 | Do Not Call | string | 0 | 0.00% | 0 | 0.00% |
| 7 | Converted (Target) | int64 | 0 | 0.00% | 0 | 0.00% |
| 8 | TotalVisits | float64 | 137 | 1.48% | 0 | 1.48% |
| 9 | Total Time Spent on Website | int64 | 0 | 0.00% | 0 | 0.00% |
| 10 | Page Views Per Visit | float64 | 137 | 1.48% | 0 | 1.48% |
| 11 | Last Activity | string | 103 | 1.11% | 0 | 1.11% |
| 12 | Country | string | 2,461 | 26.63% | 0 | 26.63% |
| 13 | Specialization | string | 1,438 | 15.56% | 1,942 | 36.58% |
| 14 | How did you hear about X Education | string | 2,207 | 23.89% | 5,043 | 78.46% |
| 15 | What is your current occupation | string | 2,690 | 29.11% | 0 | 29.11% |
| 16 | What matters most to you in choosing a course | string | 2,709 | 29.32% | 0 | 29.32% |
| 17 | Search | string | 0 | 0.00% | 0 | 0.00% |
| 18 | Magazine | string | 0 | 0.00% | 0 | 0.00% |
| 19 | Newspaper Article | string | 0 | 0.00% | 0 | 0.00% |
| 20 | X Education Forums | string | 0 | 0.00% | 0 | 0.00% |
| 21 | Newspaper | string | 0 | 0.00% | 0 | 0.00% |
| 22 | Digital Advertisement | string | 0 | 0.00% | 0 | 0.00% |
| 23 | Through Recommendations | string | 0 | 0.00% | 0 | 0.00% |
| 24 | Receive More Updates About Our Courses | string | 0 | 0.00% | 0 | 0.00% |
| 25 | Tags | string | 3,353 | 36.29% | 0 | 36.29% |
| 26 | Lead Quality | string | 4,767 | 51.59% | 0 | 51.59% |
| 27 | Update me on Supply Chain Content | string | 0 | 0.00% | 0 | 0.00% |
| 28 | Get updates on DM Content | string | 0 | 0.00% | 0 | 0.00% |
| 29 | Lead Profile | string | 2,709 | 29.32% | 4,146 | 74.19% |
| 30 | City | string | 1,420 | 15.37% | 2,249 | 39.71% |
| 31 | Asymmetrique Activity Index | string | 4,218 | 45.65% | 0 | 45.65% |
| 32 | Asymmetrique Profile Index | string | 4,218 | 45.65% | 0 | 45.65% |
| 33 | Asymmetrique Activity Score | float64 | 4,218 | 45.65% | 0 | 45.65% |
| 34 | Asymmetrique Profile Score | float64 | 4,218 | 45.65% | 0 | 45.65% |
| 35 | I agree to pay the amount through cheque | string | 0 | 0.00% | 0 | 0.00% |
| 36 | A free copy of Mastering The Interview | string | 0 | 0.00% | 0 | 0.00% |
| 37 | Last Notable Activity | string | 0 | 0.00% | 0 | 0.00% |

---

## 3. Detailed Data Findings

### 3.1 Placeholder Values ("Select")
In multiple form fields, the default unselected dropdown option was recorded as the string `"Select"`. This is functionally equivalent to an unselected or missing value:
- `How did you hear about X Education`: 5,043 occurrences (54.58%) + 2,207 NaNs = **78.46% unusable**
- `Lead Profile`: 4,146 occurrences (44.87%) + 2,709 NaNs = **74.19% unusable**
- `City`: 2,249 occurrences (24.34%) + 1,420 NaNs = **39.71% unusable**
- `Specialization`: 1,942 occurrences (21.02%) + 1,438 NaNs = **36.58% unselected/missing**

### 3.2 Zero-Variance (Constant) Features
The following 5 columns contain exactly **one single unique value (`"No"`)** across all 9,240 records:
1. `Magazine` (100% "No")
2. `Receive More Updates About Our Courses` (100% "No")
3. `Update me on Supply Chain Content` (100% "No")
4. `Get updates on DM Content` (100% "No")
5. `I agree to pay the amount through cheque` (100% "No")

*Conclusion*: These columns carry zero informational entropy and cannot be used in modeling.

### 3.3 Near-Zero Variance Features (<0.2% positive class)
Several promotional/marketing channels have near-zero positive variance:
- `Do Not Call`: 9,238 "No", 2 "Yes" (99.98% "No")
- `Newspaper Article`: 9,238 "No", 2 "Yes" (99.98% "No")
- `X Education Forums`: 9,239 "No", 1 "Yes" (99.99% "No")
- `Newspaper`: 9,239 "No", 1 "Yes" (99.99% "No")
- `Digital Advertisement`: 9,236 "No", 4 "Yes" (99.96% "No")
- `Through Recommendations`: 9,233 "No", 7 "Yes" (99.92% "No")
- `Search`: 9,226 "No", 14 "Yes" (99.85% "No")
- `What matters most to you in choosing a course`: Out of 6,531 non-nulls, 6,528 are "Better Career Prospects" (99.95%).

*Conclusion*: Inclusion of these near-constant features risks severe collinearity, singular matrix inversion errors, and spurious model splits.

### 3.4 Numerical Features & Outlier Analysis

| Feature | Min | Q1 (25%) | Median | Q3 (75%) | 95th Pct | 99th Pct | Max | Mean | Std | Missing Count |
|---|---|---|---|---|---|---|---|---|---|---|
| `TotalVisits` | 0.0 | 1.0 | 3.0 | 5.0 | 10.0 | 17.0 | **251.0** | 3.45 | 4.85 | 137 (1.48%) |
| `Total Time Spent on Website` | 0 | 12 | 248 | 936 | 1562 | 1841 | **2272** | 487.70 | 548.02 | 0 (0.00%) |
| `Page Views Per Visit` | 0.0 | 1.0 | 2.0 | 3.0 | 6.0 | 9.0 | **55.0** | 2.36 | 2.16 | 137 (1.48%) |

- `TotalVisits` has extreme upper outliers (max = 251 while 99% of values are $\le 17$).
- `Page Views Per Visit` has extreme upper outliers (max = 55 while 99% of values are $\le 9$).
- `Total Time Spent on Website` is well distributed between 0 and 2,272 seconds (~38 minutes max session) without pathological artifacts.
- Outlier mitigation strategy: Robust percentile clipping (e.g. 99th percentile cap) during preprocessing rather than discarding valid rows.

### 3.5 Target Leakage and Sales-Stage Lookahead Risks

1. **`Tags` (CRITICAL LEAKAGE)**:
   - Contains values like `"Closed by Horizzon"` (99.44% converted) and `"Will revert after reading the email"` (96.86% converted) alongside `"Ringing"` (97.17% not converted).
   - These tags are assigned by sales agents *during or after* phone/email outreach. A newly arriving lead cannot have these tags.
   - **Action**: Must be excluded from all predictive models to prevent catastrophic data leakage.

2. **`Lead Quality` (CRITICAL LEAKAGE & 51.6% Missing)**:
   - Subjective score assigned manually by sales reps during calling ("High in Relevance" = 94.66% conversion).
   - **Action**: Excluded from model features.

3. **`Last Activity` & `Last Notable Activity` (ACTIVITY LEAKAGE)**:
   - Includes post-interaction events such as `"Converted to Lead"` (428 rows), `"Email Opened"`, `"SMS Sent"`, and `"Email Bounced"`.
   - In a production environment predicting probability upon lead capture, outbound marketing interaction logs have not yet unfolded.
   - **Action**: Excluded from initial baseline conversion model.

4. **`Asymmetrique` Index & Scores (45.65% Missing)**:
   - External third-party consulting scores with nearly half the records missing.
   - **Action**: Excluded due to excessive missingness and external scoring dependencies.

---

## 4. Phase 1 Audit Conclusion
The raw dataset is structurally intact, has no duplicate entities, and has a reliable binary target distribution (38.54% conversion rate). All data quality hurdles (placeholder handling, leakage columns, zero variance) have been identified, quantified, and isolated.
