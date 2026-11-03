"""
Page 1: Lead Analytics Dashboard
Interactive exploration of historical lead conversion patterns with dynamic slicers.
Includes location analysis, lead quality breakdown, and explicit data values on all charts.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_cleaning import load_cleaned_data

# Clean Dark Theme Plot Styling
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.autolayout": True,
    "figure.facecolor": "#1E293B",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#CBD5E1",
    "xtick.color": "#CBD5E1",
    "ytick.color": "#CBD5E1",
    "text.color": "#F8FAFC",
    "grid.color": "#334155",
    "grid.alpha": 0.6,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Clean Dark Theme UI Styling
st.markdown(
    """
    <style>
    .page-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 2.15rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-num {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0.2rem 0;
    }
    .metric-sub {
        font-size: 0.82rem;
        font-weight: 600;
    }
    /* Slicer multiselect and slider styling */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F8FAFC !important;
    }
    span[data-baseweb="tag"] {
        background-color: #334155 !important;
        color: #F8FAFC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown('<div class="page-title">Interactive Lead Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Analyze conversion distributions, geographic concentration, sales lead quality, and digital engagement.</div>',
        unsafe_allow_html=True,
    )

    df = load_cleaned_data()
    total_population = len(df)
    baseline_conv_rate = round((df["Converted"].sum() / total_population) * 100, 2)

    # Sidebar Interactive Filters
    st.sidebar.markdown("### 🔍 Filter Telemetry")

    # Filter: City / Location
    all_cities = sorted(df["City"].unique().tolist())
    selected_cities = st.sidebar.multiselect("Location / City", options=all_cities, default=all_cities)

    # Filter: Lead Quality
    all_qualities = sorted(df["Lead Quality"].unique().tolist())
    selected_qualities = st.sidebar.multiselect("Lead Quality Status", options=all_qualities, default=all_qualities)

    # Filter: Lead Origin
    all_origins = sorted(df["Lead Origin"].unique().tolist())
    selected_origins = st.sidebar.multiselect("Lead Origin", options=all_origins, default=all_origins)

    # Filter: Lead Source
    all_sources = sorted(df["Lead Source"].unique().tolist())
    selected_sources = st.sidebar.multiselect("Lead Source", options=all_sources, default=all_sources)

    # Filter: Current Occupation
    all_jobs = sorted(df["What is your current occupation"].unique().tolist())
    selected_jobs = st.sidebar.multiselect("Current Job / Occupation", options=all_jobs, default=all_jobs)

    # Filter: Time Spent Slider
    min_time = int(df["Total Time Spent on Website"].min())
    max_time = int(df["Total Time Spent on Website"].max())
    time_bounds = st.sidebar.slider(
        "Time on Website (Seconds)",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        step=30,
    )

    # Apply Filters
    filt_df = df[
        (df["City"].isin(selected_cities))
        & (df["Lead Quality"].isin(selected_qualities))
        & (df["Lead Origin"].isin(selected_origins))
        & (df["Lead Source"].isin(selected_sources))
        & (df["What is your current occupation"].isin(selected_jobs))
        & (df["Total Time Spent on Website"] >= time_bounds[0])
        & (df["Total Time Spent on Website"] <= time_bounds[1])
    ].copy()

    # Dynamic KPI Cards (Total Leads, Conversion Rate, Converted Leads, Lost Leads)
    n_leads = len(filt_df)
    if n_leads > 0:
        n_conv = int(filt_df["Converted"].sum())
        n_lost = n_leads - n_conv
        curr_rate = round((n_conv / n_leads) * 100, 2)
        delta_rate = round(curr_rate - baseline_conv_rate, 2)
        delta_sign = "+" if delta_rate >= 0 else ""
        delta_color = "#059669" if delta_rate >= 0 else "#DC2626"
    else:
        n_conv = 0
        n_lost = 0
        curr_rate = 0.0
        delta_rate = 0.0
        delta_sign = ""
        delta_color = "#64748B"

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Leads</div>
                <div class="metric-num">{n_leads:,}</div>
                <div class="metric-sub" style="color: #64748B;">{(n_leads/total_population)*100:.1f}% of total data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Conversion Rate</div>
                <div class="metric-num" style="color: #2563EB;">{curr_rate}%</div>
                <div class="metric-sub" style="color: {delta_color};">{delta_sign}{delta_rate}% vs baseline ({baseline_conv_rate}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Converted Leads</div>
                <div class="metric-num" style="color: #059669;">{n_conv:,}</div>
                <div class="metric-sub" style="color: #059669;">Enrolled Students</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Lost Leads</div>
                <div class="metric-num" style="color: #DC2626;">{n_lost:,}</div>
                <div class="metric-sub" style="color: #DC2626;">Did Not Convert</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if n_leads == 0:
        st.warning("⚠️ No leads match the selected filter combination. Please broaden your slicers.")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Location Analysis & Lead Quality Analysis
    st.markdown("### 📍 Location & Quality Telemetry")
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.markdown("#### 🗺️ Which Locations Convert Leads Best?")
        loc_summary = (
            filt_df.groupby("City")["Converted"]
            .agg(Total="count", Converted="sum", Rate="mean")
            .reset_index()
        )
        loc_summary["Rate_Pct"] = (loc_summary["Rate"] * 100).round(1)
        loc_summary = loc_summary.sort_values(by="Rate_Pct", ascending=True)

        fig_loc, ax_loc = plt.subplots(figsize=(7, 4.2))
        bars = ax_loc.barh(loc_summary["City"], loc_summary["Rate_Pct"], color="#38BDF8", edgecolor="none")
        for bar, (_, row) in zip(bars, loc_summary.iterrows()):
            w = bar.get_width()
            ax_loc.text(
                w + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}% ({int(row['Converted'])}/{int(row['Total'])})",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#F8FAFC",
            )
        ax_loc.set_xlim(0, max(loc_summary["Rate_Pct"].max() + 20, 60))
        ax_loc.set_xlabel("Conversion Rate (%)")
        ax_loc.set_title("Lead Conversion Rate by Location / City", fontweight="bold", pad=10, color="#F8FAFC")
        st.pyplot(fig_loc)
        plt.close()

    with r1_col2:
        st.markdown("#### 🏷️ Lead Quality Status & Conversion Likelihood")
        qual_summary = (
            filt_df.groupby("Lead Quality")["Converted"]
            .agg(Total="count", Converted="sum", Rate="mean")
            .reset_index()
        )
        qual_summary["Rate_Pct"] = (qual_summary["Rate"] * 100).round(1)
        qual_order = ["High in Relevance", "Low in Relevance", "Might be", "Not Sure", "Not Assessed", "Worst"]
        qual_summary["sort_order"] = qual_summary["Lead Quality"].apply(
            lambda x: qual_order.index(x) if x in qual_order else 99
        )
        qual_summary = qual_summary.sort_values(by="sort_order", ascending=False)

        fig_qual, ax_qual = plt.subplots(figsize=(7, 4.2))
        colors = ["#10B981" if r >= 70 else ("#F59E0B" if r >= 30 else "#64748B") for r in qual_summary["Rate_Pct"]]
        bars = ax_qual.barh(qual_summary["Lead Quality"], qual_summary["Rate_Pct"], color=colors, edgecolor="none")
        for bar, (_, row) in zip(bars, qual_summary.iterrows()):
            w = bar.get_width()
            ax_qual.text(
                w + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}% ({int(row['Converted'])}/{int(row['Total'])})",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#F8FAFC",
            )
        ax_qual.set_xlim(0, 115)
        ax_qual.set_xlabel("Conversion Rate (%)")
        ax_qual.set_title("Observed Conversion Rate by Lead Quality Status", fontweight="bold", pad=10, color="#F8FAFC")
        st.pyplot(fig_qual)
        plt.close()

    st.markdown("---")

    # Row 2: Inflow Channels & Occupation
    st.markdown("### 📊 Channels & Demographic Breakdown")
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.markdown("#### 🏛️ Lead Origin: Volume and Conversions")
        origin_summary = (
            filt_df.groupby("Lead Origin")["Converted"]
            .agg(Total="count", Converted="sum", Rate="mean")
            .reset_index()
        )
        origin_summary["Rate_Pct"] = (origin_summary["Rate"] * 100).round(1)
        origin_summary = origin_summary.sort_values(by="Total", ascending=False)

        fig_orig, ax_orig = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(origin_summary))
        width = 0.38

        b1 = ax_orig.bar(x - width / 2, origin_summary["Total"], width, label="Total Leads", color="#475569")
        b2 = ax_orig.bar(x + width / 2, origin_summary["Converted"], width, label="Converted", color="#10B981")

        for bar in b1:
            h = bar.get_height()
            ax_orig.text(bar.get_x() + bar.get_width() / 2, h + 20, f"{int(h):,}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#CBD5E1")
        for bar, (_, row) in zip(b2, origin_summary.iterrows()):
            h = bar.get_height()
            ax_orig.text(bar.get_x() + bar.get_width() / 2, h + 20, f"{int(h):,} ({row['Rate_Pct']}%)", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#34D399")

        ax_orig.set_xticks(x)
        ax_orig.set_xticklabels(origin_summary["Lead Origin"], rotation=15, ha="right")
        ax_orig.set_ylabel("Number of Leads")
        ax_orig.set_title("Volume vs Conversions by Lead Origin", fontweight="bold", pad=10, color="#F8FAFC")
        ax_orig.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")
        ax_orig.set_ylim(0, origin_summary["Total"].max() * 1.18)
        st.pyplot(fig_orig)
        plt.close()

    with r2_col2:
        st.markdown("#### 💼 Conversion Rate by Occupation")
        occ_perf = (
            filt_df.groupby("What is your current occupation")["Converted"]
            .agg(Total="count", Converted="sum", Rate="mean")
            .reset_index()
        )
        occ_perf["Rate_Pct"] = (occ_perf["Rate"] * 100).round(1)
        occ_perf = occ_perf.sort_values(by="Rate_Pct", ascending=True)

        fig_occ, ax_occ = plt.subplots(figsize=(7, 4.2))
        occ_bars = ax_occ.barh(occ_perf["What is your current occupation"], occ_perf["Rate_Pct"], color="#34D399", edgecolor="none")
        for bar, (_, row) in zip(occ_bars, occ_perf.iterrows()):
            w = bar.get_width()
            ax_occ.text(
                w + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}% ({int(row['Converted'])}/{int(row['Total'])})",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#F8FAFC",
            )
        ax_occ.set_xlim(0, 115)
        ax_occ.set_xlabel("Conversion Rate (%)")
        ax_occ.set_title("Conversion Rate (%) by Current Job", fontweight="bold", pad=10, color="#F8FAFC")
        st.pyplot(fig_occ)
        plt.close()

    st.markdown("---")

    # Row 3: Top Acquisition Sources & Engagement Dwell Time
    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        st.markdown("#### 🌐 Top Lead Sources (Volume & Conversion %)")
        src_summary = (
            filt_df.groupby("Lead Source")["Converted"]
            .agg(Total="count", Converted="sum", Rate="mean")
            .reset_index()
        )
        src_summary["Rate_Pct"] = (src_summary["Rate"] * 100).round(1)
        src_summary = src_summary.sort_values(by="Total", ascending=False).head(7)

        fig_src, ax_src = plt.subplots(figsize=(7, 4.2))
        x_src = np.arange(len(src_summary))
        b_src_tot = ax_src.bar(x_src - width / 2, src_summary["Total"], width, label="Total Leads", color="#475569")
        b_src_cnv = ax_src.bar(x_src + width / 2, src_summary["Converted"], width, label="Converted", color="#38BDF8")

        for bar in b_src_tot:
            h = bar.get_height()
            ax_src.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{int(h):,}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#CBD5E1")
        for bar, (_, row) in zip(b_src_cnv, src_summary.iterrows()):
            h = bar.get_height()
            ax_src.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{row['Rate_Pct']}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#60A5FA")

        ax_src.set_xticks(x_src)
        ax_src.set_xticklabels(src_summary["Lead Source"], rotation=25, ha="right")
        ax_src.set_ylabel("Count of Leads")
        ax_src.set_title("Top Acquisition Channels (Volume & Rate)", fontweight="bold", pad=10, color="#F8FAFC")
        ax_src.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")
        ax_src.set_ylim(0, src_summary["Total"].max() * 1.2)
        st.pyplot(fig_src)
        plt.close()

    with r3_col2:
        st.markdown("#### ⏱️ Digital Dwell Time Comparison")
        time_stats = filt_df.groupby("Converted")["Total Time Spent on Website"].agg(["mean", "median", "count"]).reset_index()

        fig_time, ax_time = plt.subplots(figsize=(7, 4.2))
        labels = ["Did Not Convert (0)", "Converted (1)"]
        colors = ["#475569", "#10B981"]
        bars = ax_time.bar(labels, time_stats["mean"], color=colors, width=0.45)
        for bar, (_, row) in zip(bars, time_stats.iterrows()):
            h = bar.get_height()
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                h + 20,
                f"Avg: {h:.0f}s ({h/60:.1f}m)\nMedian: {row['median']:.0f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#F8FAFC",
            )
        ax_time.set_ylabel("Average Dwell Time (Seconds)")
        ax_time.set_title("Time on Website: Non-Converted vs Converted Leads", fontweight="bold", pad=10, color="#F8FAFC")
        ax_time.set_ylim(0, time_stats["mean"].max() * 1.25)
        st.pyplot(fig_time)
        plt.close()
        ax_time.set_ylabel("Average Dwell Time (Seconds)")
        ax_time.set_title("Time on Website: Non-Converted vs Converted Leads", fontweight="bold", pad=10)
        ax_time.set_ylim(0, time_stats["mean"].max() * 1.25)
        st.pyplot(fig_time)
        plt.close()

    st.markdown("---")

    # Interactive Leads Table with City & Lead Quality
    st.markdown("### 📋 Segment Lead Inspection (Top 25 Records)")
    display_cols = [
        "Lead Number",
        "City",
        "Lead Quality",
        "Lead Origin",
        "Lead Source",
        "What is your current occupation",
        "Total Time Spent on Website",
        "TotalVisits",
        "Converted",
    ]
    st.dataframe(filt_df[display_cols].head(25), use_container_width=True)


if __name__ == "__main__":
    main()
