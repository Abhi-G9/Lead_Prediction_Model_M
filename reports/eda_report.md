# Phase 2: Exploratory Data Analysis (EDA) Report

**Project**: Lead Intelligence – Web-Based Lead Conversion Prediction Platform  
**Dataset**: Processed X Education Leads Data (`data/processed/cleaned_leads.csv`)  
**Scope**: Empirical Analysis of Conversion Drivers, Volume Channels, and Behavioral Patterns  

---

## 1. Executive Summary & Core KPIs

An in-depth exploratory analysis was performed across all 9,240 lead records to investigate the business question:
> *"Which leads are more likely to convert, and which factors are associated with successful lead conversion?"*

| KPI Metric | Value | Business Meaning |
|---|---|---|
| **Total Leads** | **9,240** | Total incoming customer inquiries in the historical dataset. |
| **Converted Leads** | **3,561** | Inquiries that successfully enrolled in an education course. |
| **Not Converted Leads** | **5,679** | Inquiries that did not enroll. |
| **Overall Conversion Rate** | **38.54%** | Baseline conversion benchmark for sales operations. |

![Overall Target Distribution](file:///c:/Web_Based_Merge/assets/eda/fig1_target_distribution.png)

---

## 2. Inflow Channels Analysis: Lead Origin

Lead Origin identifies how the lead was created in the CRM.

| Lead Origin | Total Leads | Converted Leads | Conversion Rate (%) | Inflow Share (%) |
|---|---|---|---|---|
| **Landing Page Submission** | 4,886 | 1,768 | 36.19% | 52.88% |
| **API** | 3,580 | 1,115 | 31.15% | 38.74% |
| **Lead Add Form** | 718 | 664 | **92.48%** | 7.77% |
| **Lead Import** | 56 | 14 | 25.00% | 0.61% |

![Conversion by Lead Origin](file:///c:/Web_Based_Merge/assets/eda/fig2_conversion_by_origin.png)

### Key Insights:
1. **The Lead Add Form Powerhouse**: While only accounting for 7.77% of total inquiries, leads originating from `Lead Add Form` exhibit an extraordinary **92.48% conversion rate**. These leads represent direct staff creation or qualified phone inbound entries who already have high purchase intent.
2. **Mass Channels**: `Landing Page Submission` (36.19%) and `API` (31.15%) represent over 91% of volume, forming the bulk of sales team outreach where lead prioritization yields the greatest return on sales time.

---

## 3. Marketing Acquisition: Lead Source

Lead Source identifies the external channel through which the lead discovered the platform.

| Lead Source | Total Leads | Converted Leads | Conversion Rate (%) |
|---|---|---|---|
| **Google** | 2,873 | 1,147 | **39.92%** |
| **Direct Traffic** | 2,543 | 818 | **32.17%** |
| **Olark Chat** | 1,755 | 448 | **25.53%** |
| **Organic Search** | 1,154 | 436 | **37.78%** |
| **Reference** | 534 | 490 | **91.76%** |
| **Welingak Website** | 142 | 140 | **98.59%** |
| **Referral Sites** | 125 | 31 | 24.80% |
| **Other_Source** | 59 | 38 | 64.41% |
| **Facebook** | 55 | 13 | 23.64% |

![Conversion by Lead Source](file:///c:/Web_Based_Merge/assets/eda/fig3_conversion_by_source.png)

### Key Insights:
1. **Peer Endorsements & Strategic Partners**: Leads referred via `Reference` (91.76%) and `Welingak Website` (98.59%) convert at near-certain rates. Sales teams should expedite outreach to these leads within minutes of arrival.
2. **Search Performance**: Search channels perform solidly (`Google`: 39.92%, `Organic Search`: 37.78%), outperforming `Direct Traffic` (32.17%).
3. **Chat Leads**: `Olark Chat` leads convert at only 25.53%, indicating early-stage casual browsing or quick question-asking rather than immediate enrollment intent.

---

## 4. Professional Demographic Analysis: Current Occupation

Current occupation is one of the strongest discriminatory demographic signals.

| Current Occupation | Total Leads | Converted Leads | Conversion Rate (%) |
|---|---|---|---|
| **Unemployed** | 5,600 | 2,441 | **43.59%** |
| **Unknown (Not Provided)** | 2,690 | 370 | **13.75%** |
| **Working Professional** | 706 | 647 | **91.64%** |
| **Student** | 210 | 78 | **37.14%** |
| **Other** | 34 | 25 | **73.53%** |

![Conversion by Occupation](file:///c:/Web_Based_Merge/assets/eda/fig4_conversion_by_occupation.png)

### Key Insights:
1. **Working Professionals are Prime Customers**: A staggering **91.64%** of working professionals convert. They have disposable income and immediate career progression motivation to upskill.
2. **The "Unknown" Drop-off**: Leads who skip the occupation field convert at only **13.75%**—less than a third of the average unemployed lead (43.59%). Missing occupation on a form is a strong behavioral signal of low commitment.

---

## 5. Domain Interest: Specialization

Specializations with $\ge 100$ leads were evaluated for conversion propensity.

| Specialization | Total Leads | Converted Leads | Conversion Rate (%) |
|---|---|---|---|
| **Marketing Management** | 838 | 408 | **48.69%** |
| **Operations Management** | 503 | 238 | **47.32%** |
| **Human Resource Management** | 848 | 388 | **45.75%** |
| **Finance Management** | 976 | 436 | **44.67%** |
| **Business Administration** | 403 | 179 | **44.42%** |
| **Supply Chain Management** | 349 | 151 | **43.27%** |
| **IT Projects Management** | 366 | 140 | **38.25%** |
| **Not Specified** | 3,380 | 969 | **28.67%** |

![Conversion by Specialization](file:///c:/Web_Based_Merge/assets/eda/fig5_conversion_by_specialization.png)

### Key Insights:
- Management specializations (`Marketing`, `Operations`, `HR`, `Finance`) lead in conversion (~45% - 49%).
- In contrast, unselected/missing specialization (`Not Specified`) exhibits a significantly lower conversion rate of 28.67%.

---

## 6. Digital Footprint & Web Engagement

We analyzed behavioral analytics across website interactions:

| Metric | Non-Converted Leads (Mean / Median) | Converted Leads (Mean / Median) | Statistical Difference |
|---|---|---|---|
| **Total Time Spent on Website (sec)** | **330.4s** / 179.0s | **738.6s** / 832.0s | **+123.5% longer** for converted leads |
| **TotalVisits (capped)** | 3.22 / 3.0 visits | 3.51 / 3.0 visits | Minimal difference (+0.29 visits) |
| **Page Views Per Visit (capped)** | 2.34 / 2.0 views | 2.32 / 2.0 views | Virtually identical |

![Website Engagement Comparison](file:///c:/Web_Based_Merge/assets/eda/fig6_numerical_distributions.png)
![Correlation Heatmap](file:///c:/Web_Based_Merge/assets/eda/fig7_correlation_heatmap.png)

### Critical Behavioral Finding:
- **Time Spent is the Deciding Signal**: Converted leads spend an average of **12 minutes 18 seconds (738.6s)** on the platform versus **5 minutes 30 seconds (330.4s)** for non-converting leads.
- **Page Views alone do not indicate intent**: A visitor clicking through 5 pages in 45 seconds is far less likely to convert than a visitor reading course curriculum details on 2 pages for 15 minutes. Depth of attention, not superficial page bouncing, correlates with enrollment.

---

## 7. Conclusions for Modeling & Business Strategy

1. **Top Predictive Drivers Identified**:
   - `Total Time Spent on Website` (continuous engagement)
   - `Lead Origin: Lead Add Form` (high-intent acquisition)
   - `Lead Source: Reference & Welingak Website` (referral trust)
   - `Occupation: Working Professional` (purchasing power & urgency)
   - `Occupation: Unknown` (negative indicator of interest)
2. **Actionable Sales Recommendation**:
   - Automatically route leads with $\ge 8$ minutes on site, or from `Lead Add Form` / `Working Professional` cohorts, into an instant priority follow-up queue.
