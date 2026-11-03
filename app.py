"""
Lead Intelligence – Web-Based Lead Conversion Prediction Platform
Entry Point & Navigation Manager
Configures the 2-page application:
1. Lead Analytics
2. Lead Conversion Prediction
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Lead Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme Sidebar & UI Styling
st.markdown(
    """
    <style>
    /* Dark Surface Backgrounds */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }
    /* Slicer & Input Dark Styling */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F8FAFC !important;
    }
    span[data-baseweb="tag"] {
        background-color: #334155 !important;
        color: #F8FAFC !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        color: #FFFFFF;
        border-color: #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.8rem 0 1rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 1rem;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF !important; letter-spacing: -0.01em;">
                🎯 Lead Intelligence
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;">
                Machine Learning Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clean 2-Page Navigation
page_analytics = st.Page(
    "pages/1_Lead_Analytics.py",
    title="Lead Analytics",
    icon="📈",
    default=True,
)

page_prediction = st.Page(
    "pages/2_Lead_Prediction.py",
    title="Lead Conversion Prediction",
    icon="🎯",
)

pg = st.navigation([page_analytics, page_prediction])
pg.run()
