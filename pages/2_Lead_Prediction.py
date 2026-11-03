"""
Page 2: Lead Conversion Prediction
Interactive lead scoring engine using the persisted Scikit-Learn pipeline.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.predict import predict_lead

# Dark Theme UI Styling
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
        line-height: 1.5;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.6rem;
    }
    .prediction-card {
        padding: 1.6rem 1.8rem;
        border-radius: 12px;
        margin-top: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .high-priority {
        background-color: #064E3B;
        border: 2px solid #10B981;
    }
    .medium-priority {
        background-color: #78350F;
        border: 2px solid #F59E0B;
    }
    .low-priority {
        background-color: #1E293B;
        border: 2px solid #64748B;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-high { background-color: #10B981; color: #FFFFFF; }
    .badge-med { background-color: #F59E0B; color: #FFFFFF; }
    .badge-low { background-color: #64748B; color: #FFFFFF; }
    .stat-hero {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
        margin: 0.4rem 0;
    }
    .action-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    /* Slicer multiselect, dropdown, and inputs in dark theme */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="input"] > div {
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
    # Page Header
    st.markdown('<div class="page-title">Lead Conversion Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Enter the lead details to estimate the chance of conversion.</div>',
        unsafe_allow_html=True,
    )

    # Quick Scenario Presets for presentation convenience
    st.markdown("##### 📌 Load Representative Lead Profiles")
    preset_cols = st.columns(3)

    if preset_cols[0].button("🌟 Scenario: High-Intent Working Professional", use_container_width=True):
        st.session_state["p_origin"] = "Lead Add Form"
        st.session_state["p_source"] = "Reference"
        st.session_state["p_job"] = "Working Professional"
        st.session_state["p_study"] = "Finance Management"
        st.session_state["p_email_update"] = "Yes"
        st.session_state["p_guide"] = "Yes"
        st.session_state["p_visits"] = 6
        st.session_state["p_time"] = 1250
        st.session_state["p_pv"] = 3.5

    if preset_cols[1].button("⚖️ Scenario: Standard Web Inquiry", use_container_width=True):
        st.session_state["p_origin"] = "Landing Page Submission"
        st.session_state["p_source"] = "Google"
        st.session_state["p_job"] = "Unemployed"
        st.session_state["p_study"] = "Marketing Management"
        st.session_state["p_email_update"] = "Yes"
        st.session_state["p_guide"] = "No"
        st.session_state["p_visits"] = 3
        st.session_state["p_time"] = 420
        st.session_state["p_pv"] = 2.0

    if preset_cols[2].button("❄️ Scenario: Cold Form Drop-off", use_container_width=True):
        st.session_state["p_origin"] = "API"
        st.session_state["p_source"] = "Olark Chat"
        st.session_state["p_job"] = "Unknown"
        st.session_state["p_study"] = "Not Specified"
        st.session_state["p_email_update"] = "No"
        st.session_state["p_guide"] = "No"
        st.session_state["p_visits"] = 1
        st.session_state["p_time"] = 35
        st.session_state["p_pv"] = 1.0

    st.markdown("<br>", unsafe_allow_html=True)

    # Lead Input Form
    with st.form("lead_conversion_form"):
        # Section 1: Lead Information
        st.markdown(
            '<div class="section-header">1. Lead Information</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            origin_opts = ["Landing Page Submission", "API", "Lead Add Form", "Lead Import"]
            cur_origin = st.session_state.get("p_origin", "Landing Page Submission")
            idx_origin = origin_opts.index(cur_origin) if cur_origin in origin_opts else 0
            lead_origin = st.selectbox(
                "Lead Origin",
                options=origin_opts,
                index=idx_origin,
                help="How the lead entered the CRM (e.g. Landing Page, API, Direct Lead Add Form)",
            )

            source_opts = [
                "Google",
                "Direct Traffic",
                "Olark Chat",
                "Organic Search",
                "Reference",
                "Welingak Website",
                "Referral Sites",
                "Facebook",
                "Other_Source",
            ]
            cur_source = st.session_state.get("p_source", "Google")
            idx_source = source_opts.index(cur_source) if cur_source in source_opts else 0
            lead_source = st.selectbox(
                "Lead Source",
                options=source_opts,
                index=idx_source,
                help="Original marketing channel (Google, Organic Search, Reference, etc.)",
            )

        with c2:
            job_opts = ["Working Professional", "Unemployed", "Student", "Unknown", "Other"]
            cur_job = st.session_state.get("p_job", "Unemployed")
            idx_job = job_opts.index(cur_job) if cur_job in job_opts else 1
            current_job = st.selectbox(
                "Current Job",
                options=job_opts,
                index=idx_job,
                help="The lead's declared employment status",
            )

            study_opts = [
                "Finance Management",
                "Human Resource Management",
                "Marketing Management",
                "Operations Management",
                "Business Administration",
                "IT Projects Management",
                "Supply Chain Management",
                "Banking, Investment And Insurance",
                "Media and Advertising",
                "Travel and Tourism",
                "International Business",
                "Healthcare Management",
                "Hospitality Management",
                "E-Commerce",
                "Retail Management",
                "Rural and Agribusiness",
                "E-Business",
                "Services Excellence",
                "Not Specified",
            ]
            cur_study = st.session_state.get("p_study", "Marketing Management")
            idx_study = study_opts.index(cur_study) if cur_study in study_opts else 2
            area_of_study = st.selectbox(
                "Area of Study",
                options=study_opts,
                index=idx_study,
                help="Course specialization or functional domain interest",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 2: Contact Preferences
        st.markdown(
            '<div class="section-header">2. Contact Preferences</div>',
            unsafe_allow_html=True,
        )
        p1, p2 = st.columns(2)
        with p1:
            email_opts = ["Yes", "No"]
            cur_email = st.session_state.get("p_email_update", "Yes")
            idx_email = email_opts.index(cur_email) if cur_email in email_opts else 0
            wants_email = st.selectbox(
                "Does the lead want email updates?",
                options=email_opts,
                index=idx_email,
                help="Select 'Yes' if lead is receptive to email; 'No' if opted out",
            )

        with p2:
            guide_opts = ["No", "Yes"]
            cur_guide = st.session_state.get("p_guide", "No")
            idx_guide = guide_opts.index(cur_guide) if cur_guide in guide_opts else 0
            requested_guide = st.selectbox(
                "Requested free interview guide?",
                options=guide_opts,
                index=idx_guide,
                help="Whether lead checked the box for 'Mastering The Interview' handbook",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 3: Website Activity
        st.markdown(
            '<div class="section-header">3. Website Activity</div>',
            unsafe_allow_html=True,
        )
        w1, w2, w3 = st.columns(3)
        with w1:
            total_visits = st.number_input(
                "Total Website Visits",
                min_value=0,
                max_value=250,
                value=int(st.session_state.get("p_visits", 3)),
                step=1,
                help="Total distinct browsing sessions recorded",
            )
        with w2:
            time_spent = st.number_input(
                "Time on Website (Seconds)",
                min_value=0,
                max_value=3000,
                value=int(st.session_state.get("p_time", 350)),
                step=15,
                help="Cumulative active dwell duration across visits (e.g., 300s = 5 mins, 720s = 12 mins)",
            )
            st.caption(f"Equivalent duration: **{time_spent/60:.1f} minutes**")
        with w3:
            pages_viewed = st.number_input(
                "Pages Viewed Per Visit",
                min_value=0.0,
                max_value=60.0,
                value=float(st.session_state.get("p_pv", 2.0)),
                step=0.5,
                help="Average page count explored per session",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button(
            "⚡ Predict Conversion",
            use_container_width=True,
            type="primary",
        )

    # Prediction Outcome Processing
    if predict_btn:
        # Map user-friendly form fields to model input features
        do_not_email_val = "No" if wants_email == "Yes" else "Yes"

        input_payload = {
            "TotalVisits": float(total_visits),
            "Total Time Spent on Website": int(time_spent),
            "Page Views Per Visit": float(pages_viewed),
            "Lead Origin": lead_origin,
            "Lead Source": lead_source,
            "Do Not Email": do_not_email_val,
            "Specialization": area_of_study,
            "What is your current occupation": current_job,
            "A free copy of Mastering The Interview": requested_guide,
        }

        try:
            res = predict_lead(input_payload)
        except Exception as e:
            st.error(f"Inference pipeline execution error: {e}")
            return

        prob = res["conversion_probability"]
        prob_pct = res["conversion_probability_pct"]
        tier = res["priority_tier"]

        if tier == "High Priority":
            card_class = "high-priority"
            badge_class = "badge-high"
        elif tier == "Medium Priority":
            card_class = "medium-priority"
            badge_class = "badge-med"
        else:
            card_class = "low-priority"
            badge_class = "badge-low"

        st.markdown("### 📊 Prediction Output")
        st.markdown("Get estimated probability and follow-up priority.")

        st.markdown(
            f"""
            <div class="prediction-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <span class="badge-pill {badge_class}">{tier}</span>
                        <div class="stat-hero">{prob_pct}%</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                            Classification Outcome: {res['predicted_label']}
                        </div>
                    </div>
                </div>
                <div style="margin-top: 1rem; margin-bottom: 0.8rem;">
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.3rem;">Probability Gauge</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(prob)

        # Operational Follow-Up Guidance
        st.markdown("#### 🎯 Follow-Up Priority & Sales Guidance")
        guidance_cols = st.columns([1.5, 1])

        with guidance_cols[0]:
            st.markdown(
                f"""
                <div class="action-card">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.3rem;">Recommended Sales Action</div>
                    <div style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">{res['recommendation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with guidance_cols[1]:
            st.markdown(
                f"""
                <div class="action-card">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.3rem;">Profile Summary</div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.6;">
                        • <strong>Job:</strong> {current_job}<br>
                        • <strong>Channel:</strong> {lead_origin} ({lead_source})<br>
                        • <strong>Engagement:</strong> {time_spent}s ({time_spent/60:.1f}m), {total_visits} visits
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
