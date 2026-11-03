"""
Data Audit Module
Performs Phase 1 comprehensive audit on the raw X Education Leads dataset.
Generates:
- reports/missing_values_report.csv
- reports/data_audit_report.md
- reports/column_selection_report.md
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data


def run_data_audit() -> dict:
    """
    Executes a comprehensive data audit on the raw dataset.
    Returns a dictionary of audit findings.
    """
    df = load_raw_data()
    n_rows, n_cols = df.shape

    # 1. Column Names and Types
    col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Missing values & 'Select' placeholders
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / n_rows) * 100

    # Count 'Select' placeholder occurrences per column
    select_counts = {}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Count string matches for 'Select' case-insensitively or exact
            matches = (df[col].astype(str).str.strip().str.lower() == "select").sum()
            select_counts[col] = int(matches)
        else:
            select_counts[col] = 0

    total_unusable = {}
    for col in df.columns:
        total_unusable[col] = int(missing_counts[col] + select_counts[col])

    missing_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data_Type": [col_dtypes[c] for c in df.columns],
            "Null_Count": [int(missing_counts[c]) for c in df.columns],
            "Null_Percentage": [round(missing_pcts[c], 2) for c in df.columns],
            "Select_Placeholder_Count": [select_counts[c] for c in df.columns],
            "Select_Percentage": [
                round((select_counts[c] / n_rows) * 100, 2) for c in df.columns
            ],
            "Total_Missing_Or_Select_Count": [total_unusable[c] for c in df.columns],
            "Total_Missing_Or_Select_Pct": [
                round((total_unusable[c] / n_rows) * 100, 2) for c in df.columns
            ],
            "Unique_Values_Count": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )

    # Save missing_values_report.csv
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(reports_dir / "missing_values_report.csv", index=False)

    # 3. Duplicate checks
    duplicate_rows_full = int(df.duplicated().sum())
    duplicate_prospect_id = int(df["Prospect ID"].duplicated().sum()) if "Prospect ID" in df.columns else 0
    duplicate_lead_number = int(df["Lead Number"].duplicated().sum()) if "Lead Number" in df.columns else 0

    # 4. Target variable analysis
    target_col = "Converted"
    target_stats = {}
    if target_col in df.columns:
        target_nulls = int(df[target_col].isna().sum())
        target_counts = df[target_col].value_counts(dropna=False).to_dict()
        target_pcts = (
            (df[target_col].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
            .to_dict()
        )
        unique_targets = list(df[target_col].unique())
        target_stats = {
            "null_count": target_nulls,
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "percentages": {str(k): float(v) for k, v in target_pcts.items()},
            "unique_values": [str(x) for x in unique_targets],
            "is_binary": set(unique_targets).issubset({0, 1}),
        }

    # 5. Numerical features audit
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_summary = {}
    for col in num_cols:
        num_summary[col] = {
            "min": float(df[col].min()),
            "q25": float(df[col].quantile(0.25)),
            "median": float(df[col].median()),
            "q75": float(df[col].quantile(0.75)),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "negative_count": int((df[col] < 0).sum()),
            "null_count": int(df[col].isna().sum()),
        }

    # 6. Categorical features audit (sample top 5 categories)
    cat_cols = [
        c
        for c in df.columns
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
    ]
    cat_summary = {}
    for col in cat_cols:
        freq = df[col].value_counts(dropna=False).head(5).to_dict()
        cat_summary[col] = {
            "nunique": int(df[col].nunique(dropna=False)),
            "top_5": {str(k): int(v) for k, v in freq.items()},
        }

    audit_results = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows_full,
        "duplicate_prospect_ids": duplicate_prospect_id,
        "duplicate_lead_numbers": duplicate_lead_number,
        "target_stats": target_stats,
        "num_summary": num_summary,
        "cat_summary": cat_summary,
        "columns": df.columns.tolist(),
    }

    return audit_results


if __name__ == "__main__":
    results = run_data_audit()
    print("Audit Complete!")
    print(f"Dataset Shape: {results['n_rows']} rows, {results['n_cols']} columns")
    print(f"Target Stats: {results['target_stats']}")
    print(f"Duplicates: {results['duplicate_rows']} full rows, {results['duplicate_prospect_ids']} prospect IDs")
