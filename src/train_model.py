"""
Model Training & Cross-Validation Module
Builds, tunes, cross-validates, and evaluates the 4 required Scikit-Learn models:
1. Dummy Classifier (Baseline)
2. Logistic Regression
3. Decision Tree Classifier
4. Random Forest Classifier

Guarantees:
- Preprocessing fitted on training data only
- Stratified 80:20 train-test split
- 5-Fold Stratified Cross-Validation on training data
- Model persistence via Joblib
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

from src.data_cleaning import load_cleaned_data
from src.feature_engineering import (
    create_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_candidate_models() -> dict[str, object]:
    """
    Instantiates the 4 required algorithms with balanced/regularized parameters.
    """
    return {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            C=0.5,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=15,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_evaluate_all_models() -> dict:
    """
    Executes the full model training, cross-validation, and test evaluation workflow.
    """
    df = load_cleaned_data()

    # Features and target
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy().astype(int)

    # 1. Stratified 80:20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = build_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {}
    fitted_pipelines = {}

    for name, estimator in models.items():
        print(f"--> Training and evaluating: {name}...")

        # Construct end-to-end Pipeline
        # Preprocessor scales numerics and one-hot encodes categoricals
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(scale_numeric=True)),
                ("classifier", estimator),
            ]
        )

        # 2. 5-Fold Stratified Cross-Validation on training set
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        # 3. Fit pipeline on full training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # 4. Predict on held-out test data
        y_pred = pipeline.predict(X_test)
        
        # Probabilities
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
            brier = float(brier_score_loss(y_test, y_proba))
        else:
            y_proba = np.zeros_like(y_pred, dtype=float)
            auc = 0.5
            brier = float(brier_score_loss(y_test, y_proba))

        # Metrics on Test Set
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Summary cv scores
        cv_summary = {
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "train_accuracy_mean": float(np.mean(cv_scores["train_accuracy"])),
            "train_f1_mean": float(np.mean(cv_scores["train_f1"])),
        }

        results[name] = {
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": auc,
            "brier_score": brier,
            "confusion_matrix": cm,
            "classification_report": report,
            "cv_summary": cv_summary,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
        }

    # Identify best model based on F1-score and ROC-AUC
    # We do not use accuracy alone because Converted is 38.5%
    scored_models = [m for m in results.keys() if m != "Dummy Classifier"]
    best_model_name = max(
        scored_models,
        key=lambda m: (results[m]["test_f1"] * 0.5 + results[m]["test_roc_auc"] * 0.5),
    )
    print(f"\nOptimal Model Selected: {best_model_name}")

    best_pipeline = fitted_pipelines[best_model_name]

    # Save best pipeline
    best_pipeline_path = MODELS_DIR / "trained_pipeline.joblib"
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"Saved best pipeline to: {best_pipeline_path}")

    # Save complete artifacts dictionary for evaluation and app display
    all_artifacts = {
        "best_model_name": best_model_name,
        "results": results,
        "feature_cols": feature_cols,
        "fitted_pipelines": fitted_pipelines,
        "test_indices": X_test.index.tolist(),
    }
    all_artifacts_path = MODELS_DIR / "all_models_and_metrics.joblib"
    joblib.dump(all_artifacts, all_artifacts_path)
    print(f"Saved all models and metrics to: {all_artifacts_path}")

    return all_artifacts


if __name__ == "__main__":
    train_and_evaluate_all_models()
