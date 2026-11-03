# Phase 2: Data Cleaning & Preprocessing Report

**Project**: Lead Intelligence – Web-Based Lead Conversion Prediction Platform  
**Target Dataset**: Processed working copy saved at `data/processed/cleaned_leads.csv`  
**Raw Source**: Preserved intact at `data/original/Leads.csv`  

---

## 1. Principles & Objectives of Data Cleaning

Data cleaning in Phase 2 was conducted under rigorous data integrity constraints:
1. **Preservation of Raw Data**: The original CSV in `data/original/` was strictly preserved in its pristine state. All transformations were executed on an isolated working copy.
2. **Zero Information Loss**: No valid customer record was deleted. The working dataset retains the full **9,240 rows**.
3. **Target Leakage Neutralization**: Post-outreach sales logs (`Tags`, `Lead Quality`, `Last Activity`, `Last Notable Activity`) were excluded from model input columns.
4. **Resolution of Placeholder Values**: The string `"Select"` was properly categorized as an unselected dropdown choice rather than treated as a genuine domain label.
5. **Robust Outlier Handling**: Rather than deleting high-engagement visitors, extreme values in `TotalVisits` and `Page Views Per Visit` were clipped at the 99th percentile.

---

## 2. Transformation Log & Decisions

### 2.1 Categorical Feature Standardization

| Feature Name | Raw State Issues | Cleaning Action Applied | Processed Categories / Values |
|---|---|---|---|
| `Lead Source` | Inconsistent casing (`google` vs `Google`, `facebook` vs `Facebook`), 21 high-cardinality long-tail entries (<20 records). | Standardized case; mapped rare channels (<100 occurrences) to `'Other_Source'`. Missing values imputed as `'Unknown'`. | `Google`, `Direct Traffic`, `Olark Chat`, `Organic Search`, `Reference`, `Welingak Website`, `Referral Sites`, `Facebook`, `Other_Source` |
| `Lead Origin` | Extreme rarity: `Quick Add Form` had only 1 record. | Merged `Quick Add Form` into `Lead Import`. | `Landing Page Submission`, `API`, `Lead Add Form`, `Lead Import` |
| `Specialization` | 1,942 `"Select"` placeholders (21.0%) + 1,438 NaNs (15.6%). Total 36.6% unselected. | Mapped both `"Select"` and null values to a clear categorical indicator: `'Not Specified'`. Preserved all valid domains. | 18 specialized management/tech domains + `'Not Specified'` |
| `What is your current occupation` | 2,690 missing values (29.1%). Ultra-rare categories: `Housewife` (10) and `Businessman` (8). | Mapped missing values to `'Unknown'`. Merged low-frequency occupations into `'Other'`. | `Unemployed`, `Unknown`, `Working Professional`, `Student`, `Other` |
| `Do Not Email` | Missing or whitespace issues. | Standardized to binary strings: `'Yes'`, `'No'`. Missing imputed as `'No'`. | `'No'`, `'Yes'` |
| `A free copy of Mastering The Interview` | Form checkbox string variation. | Standardized to binary strings: `'Yes'`, `'No'`. | `'No'`, `'Yes'` |

### 2.2 Numerical Feature Imputation & Outlier Mitigation

| Feature Name | Missing Count | Imputation Method | Raw Min - Max | 99th Percentile | Cleaned Min - Max |
|---|---|---|---|---|---|
| `TotalVisits` | 137 (1.48%) | Imputed with **Median (3.0)** | 0.0 – 251.0 | 17.0 | 0.0 – 17.0 |
| `Total Time Spent on Website` | 0 (0.00%) | None required | 0 – 2,272 | 1,840.6 | 0 – 2,272 |
| `Page Views Per Visit` | 137 (1.48%) | Imputed with **Median (2.0)** | 0.0 – 55.0 | 9.0 | 0.0 – 9.0 |

*Rationale for 99th Percentile Clipping*:
In web analytics, web crawlers or hyper-active sessions can record anomalous visit counts (e.g., 251 visits). Simply deleting these records discards legitimate customer leads. Capping them at the 99th percentile (17 visits and 9 page views) stabilizes model coefficients and decision trees while maintaining complete dataset coverage.

---

## 3. Dataset Verification

- **Final Cleaned Dataset Shape**: (9,240 rows, 12 columns)
- **Null Values Remaining**: **0** across all 12 columns
- **Preserved Tracking Columns**: `Prospect ID`, `Lead Number` (retained for entity traceability)
- **Preserved Target Column**: `Converted` (100% intact, 3,561 converted / 5,679 not converted)
- **Status**: Ready for reproducible feature pipeline engineering and model training.
