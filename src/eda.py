"""
Exploratory Data Analysis (EDA) Module
Calculates conversion patterns across business dimensions and generates
publication-quality visualizations using Matplotlib and Seaborn.
Saves all charts into assets/eda/.
"""

import sys
import os
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Matplotlib writable temp cache directory
CACHE_DIR = PROJECT_ROOT / "data" / ".mplconfig"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_cleaning import load_cleaned_data


# Set professional styling for all charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
})

ASSETS_DIR = PROJECT_ROOT / "assets" / "eda"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_eda_metrics(df: pd.DataFrame) -> dict:
    """
    Computes summary metrics and conversion rates across key business dimensions.
    """
    total_leads = len(df)
    converted_leads = int(df["Converted"].sum())
    not_converted_leads = total_leads - converted_leads
    overall_conv_rate = round((converted_leads / total_leads) * 100, 2)

    # Conversion by Lead Origin
    origin_df = df.groupby("Lead Origin")["Converted"].agg(["count", "sum", "mean"]).reset_index()
    origin_df.columns = ["Lead Origin", "Total Leads", "Converted Leads", "Conversion Rate"]
    origin_df["Conversion Rate"] = (origin_df["Conversion Rate"] * 100).round(2)
    origin_df = origin_df.sort_values(by="Total Leads", ascending=False)

    # Conversion by Lead Source
    source_df = df.groupby("Lead Source")["Converted"].agg(["count", "sum", "mean"]).reset_index()
    source_df.columns = ["Lead Source", "Total Leads", "Converted Leads", "Conversion Rate"]
    source_df["Conversion Rate"] = (source_df["Conversion Rate"] * 100).round(2)
    source_df = source_df.sort_values(by="Total Leads", ascending=False)

    # Conversion by Occupation
    occ_df = df.groupby("What is your current occupation")["Converted"].agg(["count", "sum", "mean"]).reset_index()
    occ_df.columns = ["Occupation", "Total Leads", "Converted Leads", "Conversion Rate"]
    occ_df["Conversion Rate"] = (occ_df["Conversion Rate"] * 100).round(2)
    occ_df = occ_df.sort_values(by="Total Leads", ascending=False)

    # Conversion by Specialization
    spec_df = df.groupby("Specialization")["Converted"].agg(["count", "sum", "mean"]).reset_index()
    spec_df.columns = ["Specialization", "Total Leads", "Converted Leads", "Conversion Rate"]
    spec_df["Conversion Rate"] = (spec_df["Conversion Rate"] * 100).round(2)
    spec_df = spec_df.sort_values(by="Total Leads", ascending=False)

    # Numerical feature stats grouped by Converted
    num_cols = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
    num_stats = {}
    for col in num_cols:
        grouped = df.groupby("Converted")[col].agg(["mean", "median", "std"]).round(2).to_dict()
        num_stats[col] = grouped

    return {
        "total_leads": total_leads,
        "converted_leads": converted_leads,
        "not_converted_leads": not_converted_leads,
        "overall_conv_rate": overall_conv_rate,
        "origin_breakdown": origin_df,
        "source_breakdown": source_df,
        "occ_breakdown": occ_df,
        "spec_breakdown": spec_df,
        "num_stats": num_stats,
    }


def generate_eda_visualizations(df: pd.DataFrame) -> list[Path]:
    """
    Renders and exports all required EDA figures using Matplotlib and Seaborn.
    """
    saved_files = []

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Converted"].value_counts()
    labels = ["Not Converted (0)", "Converted (1)"]
    colors = ["#4A90E2", "#50E3C2"]
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="none")
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 80,
            f"{yval:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Overall Lead Conversion Distribution", fontsize=14, pad=15, fontweight="bold")
    ax.set_ylabel("Number of Leads", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    target_path = ASSETS_DIR / "fig1_target_distribution.png"
    plt.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(target_path)

    # 2. Conversion by Lead Origin
    fig, ax = plt.subplots(figsize=(9, 5))
    origin_order = df["Lead Origin"].value_counts().index
    sns.countplot(data=df, x="Lead Origin", hue="Converted", order=origin_order, palette=["#708090", "#2E8B57"], ax=ax)
    ax.set_title("Lead Count and Conversion by Lead Origin", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Lead Origin", fontsize=11)
    ax.set_ylabel("Count of Leads", fontsize=11)
    ax.legend(["Not Converted", "Converted"], title="Outcome", loc="upper right")
    plt.xticks(rotation=15, ha="right")
    origin_path = ASSETS_DIR / "fig2_conversion_by_origin.png"
    plt.savefig(origin_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(origin_path)

    # 3. Conversion by Top Lead Sources
    fig, ax = plt.subplots(figsize=(10, 5))
    source_order = df["Lead Source"].value_counts().index
    sns.countplot(data=df, x="Lead Source", hue="Converted", order=source_order, palette=["#708090", "#2E8B57"], ax=ax)
    ax.set_title("Lead Count and Conversion by Lead Source", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Lead Source", fontsize=11)
    ax.set_ylabel("Count of Leads", fontsize=11)
    ax.legend(["Not Converted", "Converted"], title="Outcome", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    source_path = ASSETS_DIR / "fig3_conversion_by_source.png"
    plt.savefig(source_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(source_path)

    # 4. Conversion Rate by Occupation
    occ_summary = df.groupby("What is your current occupation")["Converted"].agg(["count", "mean"]).reset_index()
    occ_summary["rate"] = occ_summary["mean"] * 100
    occ_summary = occ_summary.sort_values(by="rate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(occ_summary["What is your current occupation"], occ_summary["rate"], color="#2E8B57", edgecolor="none")
    for bar in bars:
        xval = bar.get_width()
        ax.text(
            xval + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{xval:.1f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Conversion Rate (%) by Current Occupation", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Conversion Rate (%)", fontsize=11)
    ax.set_ylabel("Occupation", fontsize=11)
    ax.set_xlim(0, 100)
    occ_path = ASSETS_DIR / "fig4_conversion_by_occupation.png"
    plt.savefig(occ_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(occ_path)

    # 5. Conversion Rate by Top Specializations
    spec_summary = df.groupby("Specialization")["Converted"].agg(["count", "mean"]).reset_index()
    spec_summary["rate"] = spec_summary["mean"] * 100
    # Keep specializations with >= 100 leads for meaningful statistical rate comparison
    spec_summary = spec_summary[spec_summary["count"] >= 100].sort_values(by="rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(spec_summary["Specialization"], spec_summary["rate"], color="#4A90E2", edgecolor="none")
    for bar in bars:
        xval = bar.get_width()
        ax.text(
            xval + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{xval:.1f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_title("Conversion Rate (%) by Specialization (N >= 100)", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Conversion Rate (%)", fontsize=11)
    ax.set_ylabel("Specialization", fontsize=11)
    ax.set_xlim(0, max(spec_summary["rate"]) * 1.15)
    spec_path = ASSETS_DIR / "fig5_conversion_by_specialization.png"
    plt.savefig(spec_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(spec_path)

    # 6. Website Engagement vs Conversion (Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("Total Time Spent on Website", "Time on Website (sec)", "#3498db"),
        ("TotalVisits", "Total Visits (capped)", "#2ecc71"),
        ("Page Views Per Visit", "Page Views / Visit (capped)", "#9b59b6"),
    ]
    for idx, (col, title, color) in enumerate(metrics):
        sns.boxplot(
            data=df,
            x="Converted",
            y=col,
            hue="Converted",
            legend=False,
            ax=axes[idx],
            palette=["#bdc3c7", color],
            boxprops=dict(alpha=0.8),
        )
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Converted (0 = No, 1 = Yes)", fontsize=10)
        axes[idx].set_ylabel(title, fontsize=10)
    plt.suptitle("Website Engagement Comparison: Non-Converted vs Converted Leads", fontsize=14, fontweight="bold", y=1.02)
    engagement_path = ASSETS_DIR / "fig6_numerical_distributions.png"
    plt.savefig(engagement_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(engagement_path)

    # 7. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    num_df = df[["Converted", "Total Time Spent on Website", "TotalVisits", "Page Views Per Visit"]].corr()
    sns.heatmap(num_df, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, square=True)
    ax.set_title("Pearson Correlation Heatmap", fontsize=13, pad=12, fontweight="bold")
    corr_path = ASSETS_DIR / "fig7_correlation_heatmap.png"
    plt.savefig(corr_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_files.append(corr_path)

    return saved_files


def run_eda_pipeline() -> dict:
    """
    Executes the full EDA analysis and visualization generation.
    """
    df = load_cleaned_data()
    metrics = calculate_eda_metrics(df)
    charts = generate_eda_visualizations(df)
    return {"metrics": metrics, "charts": charts}


if __name__ == "__main__":
    res = run_eda_pipeline()
    print("EDA Complete!")
    print(f"Total Leads: {res['metrics']['total_leads']}")
    print(f"Overall Conversion Rate: {res['metrics']['overall_conv_rate']}%")
    print(f"Generated {len(res['charts'])} figures in assets/eda/")
