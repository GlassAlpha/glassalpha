"""Healthcare Treatment Outcomes Dataset Generator.

This module generates synthetic healthcare data for demonstrating ML model auditing
in clinical decision support systems. The dataset simulates patient treatment
outcomes with demographic and clinical factors for regulatory compliance testing.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_healthcare_outcomes_dataset(
    n_samples: int = 15000,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic healthcare treatment outcomes dataset.

    Args:
        n_samples: Number of patient treatment records to generate
        random_state: Random seed for reproducibility
        output_path: Optional path to save the dataset

    Returns:
        DataFrame with patient treatment data and outcomes

    """
    np.random.seed(random_state)

    # Generate base features using sklearn
    X, y = make_classification(
        n_samples=n_samples,
        n_features=22,
        n_informative=15,
        n_redundant=4,
        n_clusters_per_class=1,
        class_sep=0.9,
        random_state=random_state,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "age",
        "bmi",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "cholesterol_total",
        "cholesterol_ldl",
        "cholesterol_hdl",
        "glucose_level",
        "heart_rate",
        "respiratory_rate",
        "oxygen_saturation",
        "temperature",
        "white_blood_cell_count",
        "red_blood_cell_count",
        "hemoglobin",
        "hematocrit",
        "platelet_count",
        "creatinine",
        "bun",
        "sodium",
        "potassium",
        "chloride",
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform features to realistic medical data ranges
    df = _transform_medical_features(df)
    df["treatment_outcome"] = y  # Binary: 1 = positive outcome, 0 = negative

    # Add demographic features for fairness analysis
    df = _add_medical_demographics(df, random_state)

    # Add medical metadata
    df["patient_id"] = [f"PAT_{i:08d}" for i in range(len(df))]
    df["treatment_date"] = pd.date_range(
        start="2023-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")
    df["treatment_type"] = np.random.choice(
        ["medication", "surgery", "therapy", "lifestyle"],
        size=len(df),
    )

    # Reorder columns
    cols = ["patient_id", "treatment_date", "treatment_type"] + feature_names + ["treatment_outcome"]
    df = df[cols]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved healthcare outcomes dataset to {output_path}")

    return df


def _transform_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform generated features to realistic medical data ranges."""
    # Age: 18-90 years
    df["age"] = np.clip((df["age"] * 18) + 35, 18, 90).astype(int)

    # BMI: 15-50 (underweight to obese)
    df["bmi"] = np.clip((df["bmi"] * 8.75) + 22.5, 15, 50).round(1)

    # Blood pressure: Systolic 90-200, Diastolic 60-120
    df["blood_pressure_systolic"] = np.clip((df["blood_pressure_systolic"] * 27.5) + 110, 90, 200).astype(int)
    df["blood_pressure_diastolic"] = np.clip((df["blood_pressure_diastolic"] * 15) + 70, 60, 120).astype(int)

    # Cholesterol: Total 120-350, LDL 50-250, HDL 20-100
    df["cholesterol_total"] = np.clip((df["cholesterol_total"] * 57.5) + 170, 120, 350).astype(int)
    df["cholesterol_ldl"] = np.clip((df["cholesterol_ldl"] * 50) + 80, 50, 250).astype(int)
    df["cholesterol_hdl"] = np.clip((df["cholesterol_hdl"] * 20) + 40, 20, 100).astype(int)

    # Glucose: 70-300 mg/dL
    df["glucose_level"] = np.clip((df["glucose_level"] * 57.5) + 100, 70, 300).astype(int)

    # Vital signs
    df["heart_rate"] = np.clip((df["heart_rate"] * 25) + 65, 50, 120).astype(int)
    df["respiratory_rate"] = np.clip((df["respiratory_rate"] * 5) + 12, 8, 25).astype(int)
    df["oxygen_saturation"] = np.clip((df["oxygen_saturation"] * 5) + 95, 90, 100).astype(int)
    df["temperature"] = np.clip((df["temperature"] * 2) + 97, 95, 102).round(1)

    # Blood cell counts
    df["white_blood_cell_count"] = np.clip((df["white_blood_cell_count"] * 3000) + 5500, 3000, 15000).astype(int)
    df["red_blood_cell_count"] = np.clip((df["red_blood_cell_count"] * 1.5) + 4.2, 2.5, 6.5).round(1)
    df["hemoglobin"] = np.clip((df["hemoglobin"] * 4) + 12, 8, 18).round(1)
    df["hematocrit"] = np.clip((df["hematocrit"] * 10) + 36, 25, 55).round(1)
    df["platelet_count"] = np.clip((df["platelet_count"] * 100000) + 200000, 100000, 500000).astype(int)

    # Kidney function
    df["creatinine"] = np.clip((df["creatinine"] * 2) + 0.8, 0.5, 4.0).round(2)
    df["bun"] = np.clip((df["bun"] * 10) + 10, 5, 30).astype(int)

    # Electrolytes
    df["sodium"] = np.clip((df["sodium"] * 10) + 135, 125, 150).astype(int)
    df["potassium"] = np.clip((df["potassium"] * 1) + 3.8, 2.5, 5.5).round(1)
    df["chloride"] = np.clip((df["chloride"] * 8) + 98, 85, 115).astype(int)

    return df


def _add_medical_demographics(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add demographic features for fairness analysis in healthcare."""
    np.random.seed(random_state)

    # Gender (protected attribute)
    df["gender"] = np.random.choice(["Male", "Female"], size=len(df))

    # Age groups (protected attribute)
    conditions = [
        (df["age"] < 30),
        (df["age"] >= 30) & (df["age"] < 50),
        (df["age"] >= 50) & (df["age"] < 65),
        (df["age"] >= 65) & (df["age"] < 80),
        (df["age"] >= 80),
    ]
    choices = ["Young_Adult", "Middle_Age", "Senior", "Elderly", "Very_Elderly"]
    df["age_group"] = np.select(conditions, choices, default="Unknown")

    # Race/ethnicity (protected attribute)
    df["race_ethnicity"] = np.random.choice(
        [
            "White",
            "Black",
            "Hispanic",
            "Asian",
            "Other",
        ],
        size=len(df),
        p=[0.6, 0.13, 0.18, 0.06, 0.03],
    )

    # Insurance type (socioeconomic indicator)
    df["insurance_type"] = np.random.choice(
        [
            "Private",
            "Medicare",
            "Medicaid",
            "Uninsured",
        ],
        size=len(df),
        p=[0.5, 0.25, 0.15, 0.1],
    )

    # Geographic region
    df["geographic_region"] = np.random.choice(
        [
            "Northeast",
            "Midwest",
            "South",
            "West",
        ],
        size=len(df),
    )

    return df


def create_healthcare_outcomes_config() -> dict[str, Any]:
    """Create example configuration for healthcare outcomes audit.

    Returns:
        Configuration dictionary for YAML serialization

    """
    return {
        "audit_profile": "healthcare_outcomes",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "~/.glassalpha/data/healthcare_outcomes.csv",
            "target_column": "treatment_outcome",
            "protected_attributes": ["gender", "age_group", "race_ethnicity"],
        },
        "model": {
            "type": "xgboost",
            "params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
        },
        "explainers": {
            "strategy": "first_compatible",
            "priority": ["treeshap", "kernelshap"],
            "config": {
                "treeshap": {"max_samples": 1000},
            },
        },
        "metrics": {
            "performance": {
                "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"],
            },
            "fairness": {
                "metrics": ["demographic_parity", "equal_opportunity", "equalized_odds"],
                "config": {
                    "demographic_parity": {"threshold": 0.05},
                    "equal_opportunity": {"threshold": 0.05},
                    "equalized_odds": {"threshold": 0.05},
                },
            },
        },
        "report": {
            "template": "standard_audit",
            "styling": {"color_scheme": "professional"},
        },
    }


if __name__ == "__main__":
    # Generate and save dataset
    df = generate_healthcare_outcomes_dataset(
        n_samples=15000,
        random_state=42,
        output_path="~/.glassalpha/data/healthcare_outcomes.csv",
    )

    print("Healthcare Outcomes Dataset Generated:")
    print(f"Shape: {df.shape}")
    print(f"Treatment success rate: {df['treatment_outcome'].mean():.1%}")
    print(f"Protected groups: {df['gender'].value_counts().to_dict()}")
    print(f"Age groups: {df['age_group'].value_counts().to_dict()}")
    print(f"Race/ethnicity: {df['race_ethnicity'].value_counts().to_dict()}")
    print(f"Treatment types: {df['treatment_type'].value_counts().to_dict()}")
