"""
Inference & Prediction Module
Provides a unified interface for lead conversion probability estimation
using the persisted Scikit-Learn pipeline.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from src.feature_engineering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

PIPELINE_PATH = PROJECT_ROOT / "models" / "trained_pipeline.joblib"

_CACHED_PIPELINE = None


def load_inference_pipeline():
    """Loads and caches the trained production pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not PIPELINE_PATH.exists():
            from src.train_model import train_and_evaluate_all_models
            train_and_evaluate_all_models()
        _CACHED_PIPELINE = joblib.load(PIPELINE_PATH)
    return _CACHED_PIPELINE


def predict_lead(lead_input: dict | pd.DataFrame) -> dict:
    """
    Predicts conversion outcome and calibrated probability for single or batch lead records.

    Args:
        lead_input: A dictionary of feature key-values or a pandas DataFrame.

    Returns:
        dict: Prediction results including probabilities, classification, priority tiers,
              and mandatory business disclaimers.
    """
    pipeline = load_inference_pipeline()

    if isinstance(lead_input, dict):
        df_input = pd.DataFrame([lead_input])
    elif isinstance(lead_input, pd.DataFrame):
        df_input = lead_input.copy()
    else:
        raise TypeError(f"Expected dict or DataFrame, received {type(lead_input)}")

    # Ensure required columns are present; fill missing with sensible defaults
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for col in expected_cols:
        if col not in df_input.columns:
            if col in NUMERICAL_FEATURES:
                df_input[col] = 0.0
            else:
                df_input[col] = "Unknown"

    # Restrict to model columns in correct order
    df_eval = df_input[expected_cols]

    # Model inference
    pred_classes = pipeline.predict(df_eval)
    pred_probas = pipeline.predict_proba(df_eval)[:, 1]

    results = []
    for cls_val, prob_val in zip(pred_classes, pred_probas):
        prob_pct = round(float(prob_val) * 100, 2)
        
        # Segment into actionable sales tiers
        if prob_val >= 0.70:
            tier = "High Priority"
            tier_color = "green"
            rec = "Immediate follow-up recommended (within 1 hour). High likelihood of conversion."
        elif prob_val >= 0.40:
            tier = "Medium Priority"
            tier_color = "orange"
            rec = "Standard nurturing queue. Provide tailored course curriculum and engagement."
        else:
            tier = "Low Priority"
            tier_color = "gray"
            rec = "Automated email drip campaign. Deprioritize manual phone outreach."

        results.append({
            "predicted_class": int(cls_val),
            "predicted_label": "Likely to Convert" if cls_val == 1 else "Unlikely to Convert",
            "conversion_probability": float(prob_val),
            "conversion_probability_pct": prob_pct,
            "priority_tier": tier,
            "tier_color": tier_color,
            "recommendation": rec,
            "disclaimer": (
                "Important Notice: This prediction is a machine learning probability estimate "
                "based on historical inquiry patterns. It does not guarantee a commercial outcome."
            ),
        })

    if isinstance(lead_input, dict):
        return results[0]
    return {"predictions": results, "summary": f"Processed {len(results)} records"}


if __name__ == "__main__":
    sample_high_intent = {
        "TotalVisits": 4.0,
        "Total Time Spent on Website": 1200,
        "Page Views Per Visit": 3.0,
        "Lead Origin": "Lead Add Form",
        "Lead Source": "Reference",
        "Do Not Email": "No",
        "Specialization": "Finance Management",
        "What is your current occupation": "Working Professional",
        "A free copy of Mastering The Interview": "No",
    }

    sample_low_intent = {
        "TotalVisits": 1.0,
        "Total Time Spent on Website": 30,
        "Page Views Per Visit": 1.0,
        "Lead Origin": "Landing Page Submission",
        "Lead Source": "Olark Chat",
        "Do Not Email": "Yes",
        "Specialization": "Not Specified",
        "What is your current occupation": "Unknown",
        "A free copy of Mastering The Interview": "No",
    }

    res_high = predict_lead(sample_high_intent)
    res_low = predict_lead(sample_low_intent)

    print("Sample High Intent Lead:")
    print(f"Prob: {res_high['conversion_probability_pct']}% | Tier: {res_high['priority_tier']}")
    print("\nSample Low Intent Lead:")
    print(f"Prob: {res_low['conversion_probability_pct']}% | Tier: {res_low['priority_tier']}")
