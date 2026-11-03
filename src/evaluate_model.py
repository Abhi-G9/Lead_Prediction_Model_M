"""
Model Evaluation & Visualization Module
Generates comprehensive evaluation charts:
- ROC Curves
- Confusion Matrix Grid
- Multi-Metric Comparison Bar Charts
- Feature Importances (Tree & Logistic Regression)
- Probability Calibration Curves (Reliability Diagram)
Saves all figures into assets/models/.
"""

import sys
import os
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Matplotlib writable cache directory
CACHE_DIR = PROJECT_ROOT / "data" / ".mplconfig"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

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

ASSETS_MODELS_DIR = PROJECT_ROOT / "assets" / "models"
ASSETS_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(artifacts: dict) -> Path:
    """Plots overlay of ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "Dummy Classifier": "#95a5a6",
        "Logistic Regression": "#3498db",
        "Decision Tree": "#e67e22",
        "Random Forest": "#2ecc71",
    }
    
    for name, res in artifacts["results"].items():
        y_test = np.array(res["y_test"])
        y_proba = np.array(res["y_proba"])
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = res["test_roc_auc"]
        linestyle = "--" if name == "Dummy Classifier" else "-"
        linewidth = 2 if name != "Dummy Classifier" else 1.5
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            color=colors.get(name, "#333333"),
            linestyle=linestyle,
            linewidth=linewidth,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    
    out_path = ASSETS_MODELS_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_confusion_matrices(artifacts: dict) -> Path:
    """Plots a 2x2 grid of confusion matrices for all evaluated models."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(artifacts["results"].items()):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label (0 = No, 1 = Converted)", fontsize=10)
        axes[idx].set_ylabel("Actual Label (0 = No, 1 = Converted)", fontsize=10)

    plt.suptitle("Confusion Matrix Comparison Across Models (Test Set: N=1,848)", fontsize=14, fontweight="bold", y=1.02)
    out_path = ASSETS_MODELS_DIR / "confusion_matrices.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_metrics_comparison(artifacts: dict) -> Path:
    """Plots grouped bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC."""
    metrics_list = []
    for name, res in artifacts["results"].items():
        metrics_list.append({
            "Model": name,
            "Accuracy": res["test_accuracy"],
            "Precision": res["test_precision"],
            "Recall": res["test_recall"],
            "F1-Score": res["test_f1"],
            "ROC-AUC": res["test_roc_auc"],
        })

    m_df = pd.DataFrame(metrics_list)
    melted = m_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=palette, ax=ax)
    ax.set_title("Test Set Evaluation Metrics Comparison", fontsize=14, pad=15, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.legend(title="Algorithm", loc="lower right", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    out_path = ASSETS_MODELS_DIR / "model_metrics_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_feature_importance(artifacts: dict) -> Path:
    """Plots top features from the Random Forest model and Logistic Regression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rf_pipe = artifacts["fitted_pipelines"]["Random Forest"]
    rf_clf = rf_pipe.named_steps["classifier"]
    preprocessor = rf_pipe.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    # Clean up feature name prefixes
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    # Random Forest Importance
    importances = rf_clf.feature_importances_
    rf_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
    rf_top = rf_df.sort_values(by="Importance", ascending=False).head(10)

    sns.barplot(data=rf_top, x="Importance", y="Feature", color="#2ecc71", ax=ax1)
    ax1.set_title("Random Forest: Top 10 Feature Importances (MDI)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Relative Importance (Gini Impurity Reduction)", fontsize=10)

    # Logistic Regression Coefficients
    lr_pipe = artifacts["fitted_pipelines"]["Logistic Regression"]
    lr_clf = lr_pipe.named_steps["classifier"]
    coefs = lr_clf.coef_[0]
    lr_df = pd.DataFrame({"Feature": clean_names, "Coefficient": coefs})
    # Top 5 positive and top 5 negative
    lr_top = pd.concat([
        lr_df.sort_values(by="Coefficient", ascending=False).head(5),
        lr_df.sort_values(by="Coefficient", ascending=True).head(5),
    ]).drop_duplicates().sort_values(by="Coefficient", ascending=False)

    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in lr_top["Coefficient"]]
    ax2.barh(lr_top["Feature"], lr_top["Coefficient"], color=colors)
    ax2.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax2.set_title("Logistic Regression: Top Predictor Coefficients", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Log-Odds Coefficient (Positive = Drives Conversion)", fontsize=10)

    plt.suptitle("Model Interpretability & Feature Drivers", fontsize=14, fontweight="bold", y=1.02)
    out_path = ASSETS_MODELS_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_calibration_curves(artifacts: dict) -> Path:
    """Plots probability calibration curves (reliability diagram) for calibrated scoring."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "Logistic Regression": "#3498db",
        "Decision Tree": "#e67e22",
        "Random Forest": "#2ecc71",
    }

    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        res = artifacts["results"][name]
        y_test = np.array(res["y_test"])
        y_proba = np.array(res["y_proba"])
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        brier = res["brier_score"]
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"{name} (Brier = {brier:.3f})",
            color=colors[name],
        )

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.6)
    ax.set_title("Probability Calibration Reliability Diagram", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Positives (Conversions)", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    out_path = ASSETS_MODELS_DIR / "calibration_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def run_model_evaluations() -> list[Path]:
    """Loads artifacts and creates all evaluation figures."""
    artifacts_path = PROJECT_ROOT / "models" / "all_models_and_metrics.joblib"
    if not artifacts_path.exists():
        from src.train_model import train_and_evaluate_all_models
        artifacts = train_and_evaluate_all_models()
    else:
        artifacts = joblib.load(artifacts_path)

    saved = [
        plot_roc_curves(artifacts),
        plot_confusion_matrices(artifacts),
        plot_metrics_comparison(artifacts),
        plot_feature_importance(artifacts),
        plot_calibration_curves(artifacts),
    ]
    print(f"Generated {len(saved)} evaluation figures in assets/models/")
    return saved


if __name__ == "__main__":
    run_model_evaluations()
