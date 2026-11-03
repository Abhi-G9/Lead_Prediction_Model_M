"""
Data Cleaning Module
Prepares a clean, leakage-free working dataset from the raw X Education Leads data.
Guarantees the raw dataset in data/original/ remains untouched.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data_loader import load_raw_data


def get_processed_data_path() -> Path:
    """Returns the standardized path to the processed dataset."""
    return PROJECT_ROOT / "data" / "processed" / "cleaned_leads.csv"


def clean_lead_sources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes casing and collapses rare lead sources into manageable categories.
    """
    df = df.copy()
    if "Lead Source" not in df.columns:
        return df

    # Replace nulls with 'Unknown'
    source = df["Lead Source"].fillna("Unknown").astype(str).str.strip()

    # Standardize casing & common duplicates
    source = source.replace({
        "google": "Google",
        "facebook": "Facebook",
        "bing": "Bing",
        "Click2call": "Other_Source",
        "Live Chat": "Other_Source",
        "NC_EDM": "Other_Source",
        "Pay per Click Ads": "Other_Source",
        "Press_Release": "Other_Source",
        "Social Media": "Other_Source",
        "WeLearn": "Other_Source",
        "blog": "Other_Source",
        "testone": "Other_Source",
        "welearnblog_Home": "Other_Source",
        "youtubechannel": "Other_Source",
    })

    # Group sources with very low representation (<100 occurrences) into 'Other_Source'
    top_sources = [
        "Google",
        "Direct Traffic",
        "Olark Chat",
        "Organic Search",
        "Reference",
        "Welingak Website",
        "Referral Sites",
        "Facebook",
    ]
    df["Lead Source"] = source.apply(lambda x: x if x in top_sources else "Other_Source")
    return df


def clean_lead_origins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates rare Lead Origins.
    """
    df = df.copy()
    if "Lead Origin" not in df.columns:
        return df

    origin = df["Lead Origin"].fillna("Unknown").astype(str).str.strip()
    # Group extremely rare origins (e.g. Quick Add Form with only 1 entry)
    origin = origin.replace({"Quick Add Form": "Lead Import"})
    df["Lead Origin"] = origin
    return df


def clean_specialization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles 'Select' placeholders and missing values in Specialization.
    Maps unselected/missing specializations to 'Not Specified'.
    """
    df = df.copy()
    if "Specialization" not in df.columns:
        return df

    spec = df["Specialization"].fillna("Not Specified").astype(str).str.strip()
    spec = spec.replace({"Select": "Not Specified", "": "Not Specified"})
    df["Specialization"] = spec
    return df


def clean_occupation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and rare categories in current occupation.
    """
    df = df.copy()
    if "What is your current occupation" not in df.columns:
        return df

    occ = df["What is your current occupation"].fillna("Unknown").astype(str).str.strip()
    # Group rare categories (<20 entries: Housewife, Businessman) into 'Other'
    occ = occ.replace({
        "Housewife": "Other",
        "Businessman": "Other",
    })
    df["What is your current occupation"] = occ
    return df


def clean_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the City column, mapping 'Select' and nulls to 'Not Specified'.
    """
    df = df.copy()
    if "City" not in df.columns:
        return df

    city = df["City"].fillna("Not Specified").astype(str).str.strip()
    city = city.replace({"Select": "Not Specified", "": "Not Specified"})
    df["City"] = city
    return df


def clean_lead_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Lead Quality column for analytical inspection.
    Maps unrated/missing records to 'Not Assessed'.
    """
    df = df.copy()
    if "Lead Quality" not in df.columns:
        return df

    quality = df["Lead Quality"].fillna("Not Assessed").astype(str).str.strip()
    quality = quality.replace({"": "Not Assessed", "Select": "Not Assessed"})
    df["Lead Quality"] = quality
    return df


def cap_numerical_outliers(
    df: pd.DataFrame,
    col: str,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Clips extreme upper outliers to a specified percentile cap (default 99th percentile).
    Does NOT drop valid records.
    """
    df = df.copy()
    if col in df.columns:
        cap_val = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower=0, upper=cap_val)
    return df


def clean_leads_data(
    raw_df: pd.DataFrame | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Full data cleaning and preparation pipeline.
    
    1. Operates on an isolated copy of the data.
    2. Imputes missing numerical values with median.
    3. Caps extreme numerical outliers at the 99th percentile without deleting rows.
    4. Cleans and standardizes categorical features (Lead Origin, Lead Source, Occupation, Specialization).
    5. Standardizes binary flags (Do Not Email, A free copy of Mastering The Interview).
    6. Drops zero-variance and severe leakage columns identified in Phase 1 audit.
    7. Retains IDs for entity lookup while distinguishing them from model inputs.
    """
    if raw_df is None:
        raw_df = load_raw_data()
    df = raw_df.copy()

    # 1. Clean categorical features
    df = clean_lead_sources(df)
    df = clean_lead_origins(df)
    df = clean_specialization(df)
    df = clean_occupation(df)
    df = clean_city(df)
    df = clean_lead_quality(df)

    # Standardize Do Not Email and A free copy of Mastering The Interview
    if "Do Not Email" in df.columns:
        df["Do Not Email"] = df["Do Not Email"].fillna("No").astype(str).str.strip()
        df["Do Not Email"] = df["Do Not Email"].map({"Yes": "Yes", "No": "No"}).fillna("No")

    if "A free copy of Mastering The Interview" in df.columns:
        df["A free copy of Mastering The Interview"] = (
            df["A free copy of Mastering The Interview"]
            .fillna("No")
            .astype(str)
            .str.strip()
        )
        df["A free copy of Mastering The Interview"] = (
            df["A free copy of Mastering The Interview"]
            .map({"Yes": "Yes", "No": "No"})
            .fillna("No")
        )

    # 2. Clean numerical features
    # TotalVisits: impute median, clip at 99th percentile
    if "TotalVisits" in df.columns:
        tv_median = df["TotalVisits"].median()
        df["TotalVisits"] = df["TotalVisits"].fillna(tv_median)
        df = cap_numerical_outliers(df, "TotalVisits", upper_quantile=0.99)

    # Page Views Per Visit: impute median, clip at 99th percentile
    if "Page Views Per Visit" in df.columns:
        pv_median = df["Page Views Per Visit"].median()
        df["Page Views Per Visit"] = df["Page Views Per Visit"].fillna(pv_median)
        df = cap_numerical_outliers(df, "Page Views Per Visit", upper_quantile=0.99)

    # Total Time Spent on Website: 0 nulls, strictly non-negative
    if "Total Time Spent on Website" in df.columns:
        df["Total Time Spent on Website"] = df["Total Time Spent on Website"].clip(lower=0)

    # 3. Columns to retain in the clean analytical dataset
    # We keep Prospect ID and Lead Number for record lookup/tracking
    # We keep the audited candidate/conditional columns suitable for analysis and modeling
    cleaned_columns = [
        "Prospect ID",
        "Lead Number",
        "Lead Origin",
        "Lead Source",
        "Do Not Email",
        "TotalVisits",
        "Total Time Spent on Website",
        "Page Views Per Visit",
        "Specialization",
        "What is your current occupation",
        "A free copy of Mastering The Interview",
        "City",
        "Lead Quality",
        "Converted",
    ]

    available_cols = [c for c in cleaned_columns if c in df.columns]
    df_clean = df[available_cols].copy()

    if save_output:
        out_path = get_processed_data_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out_path, index=False)
        print(f"Cleaned dataset saved successfully to: {out_path} ({df_clean.shape})")

    return df_clean


def load_cleaned_data() -> pd.DataFrame:
    """
    Loads the cleaned dataset. If not found, generates it via clean_leads_data().
    """
    path = get_processed_data_path()
    if not path.exists():
        return clean_leads_data(save_output=True)
    return pd.read_csv(path)


if __name__ == "__main__":
    df_cleaned = clean_leads_data()
    print("Cleaned Data Summary:")
    print(df_cleaned.info())
    print("\nMissing values in cleaned data:")
    print(df_cleaned.isna().sum())
