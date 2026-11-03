"""
Feature Engineering & Preprocessing Pipeline Module
Constructs leakage-free Scikit-Learn preprocessing transformers.
Ensures zero data leakage between training and testing splits.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Model input features
NUMERICAL_FEATURES = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
]

CATEGORICAL_FEATURES = [
    "Lead Origin",
    "Lead Source",
    "Do Not Email",
    "Specialization",
    "What is your current occupation",
    "A free copy of Mastering The Interview",
]

TARGET_COL = "Converted"
ID_COLS = ["Prospect ID", "Lead Number"]


def get_feature_names() -> dict:
    """Returns the dictionary of feature groupings."""
    return {
        "numerical": list(NUMERICAL_FEATURES),
        "categorical": list(CATEGORICAL_FEATURES),
        "target": TARGET_COL,
        "id_cols": list(ID_COLS),
    }


def create_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Creates a Scikit-Learn ColumnTransformer for numeric and categorical preprocessing.
    
    Numeric features:
    - Impute missing with median
    - Optionally scale with StandardScaler (important for Logistic Regression)
    
    Categorical features:
    - Impute missing with 'Unknown'
    - OneHotEncoder with handle_unknown='ignore' (robust to unseen categories at inference)
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drops IDs or unexpected columns
    )

    return preprocessor


if __name__ == "__main__":
    from src.data_cleaning import load_cleaned_data
    df = load_cleaned_data()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    prep = create_preprocessor()
    X_trans = prep.fit_transform(X)
    print(f"Preprocessed feature matrix shape: {X_trans.shape}")
    feature_names = prep.get_feature_names_out()
    print(f"Total encoded features: {len(feature_names)}")
