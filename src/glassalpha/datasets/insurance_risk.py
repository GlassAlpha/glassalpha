"""Insurance Risk Assessment Dataset Generator.

This module generates synthetic insurance data for demonstrating ML model auditing
in a regulated insurance context. The dataset simulates auto insurance policies
with associated risk factors and claim outcomes.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_insurance_risk_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic insurance risk assessment dataset.

    Args:
        n_samples: Number of insurance policies to generate
        random_state: Random seed for reproducibility
        output_path: Optional path to save the dataset

    Returns:
        DataFrame with insurance policy data and claim outcomes

    """
    np.random.seed(random_state)

    # Generate base features using sklearn
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=random_state,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "age",
        "annual_mileage",
        "vehicle_value",
        "deductible_amount",
        "coverage_limit",
        "years_insured",
        "previous_claims",
        "credit_score",
        "marital_status",
        "occupation_risk",
        "location_risk_score",
        "vehicle_age",
        "policy_type",
        "discount_eligibility",
        "safe_driver_discount",
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform features to realistic ranges and types
    df = _transform_features(df)
    df["claim_outcome"] = y  # Binary: 1 = filed claim, 0 = no claim

    # Add categorical demographic features for fairness analysis
    df = _add_demographic_features(df, random_state)

    # Add metadata columns
    df["policy_id"] = range(1, len(df) + 1)
    df["policy_start_date"] = pd.date_range(
        start="2023-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")

    # Reorder columns (include all columns, including demographic ones added later)
    base_cols = ["policy_id", "policy_start_date"] + feature_names + ["claim_outcome"]
    demographic_cols = ["gender", "age_group", "income_bracket"]  # Added by _add_demographic_features
    cols = base_cols + demographic_cols
    df = df[cols]

    # Perform health checks on generated dataset
    from ..data.gen_utils import assert_dataset_health  # noqa: PLC0415

    required_cols = [
        "claim_outcome",
        "age",
        "gender",
        "age_group",
        "annual_mileage",
        "vehicle_value",
        "deductible_amount",
        "coverage_limit",
        "years_insured",
        "previous_claims",
        "credit_score",
    ]

    # Note: High cardinality in policy_start_date and policy_id is expected and acceptable
    assert_dataset_health(
        df,
        expect_cols=required_cols,
        target_col="claim_outcome",
        protected_cols=["gender", "age_group"],
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved insurance dataset to {output_path}")

    return df


def _transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform generated features to realistic insurance data ranges."""
    # Age: 18-80 years
    df["age"] = np.clip((df["age"] * 15) + 35, 18, 80).astype(int)

    # Annual mileage: 0-50,000 miles
    df["annual_mileage"] = np.clip((df["annual_mileage"] * 12500) + 12500, 0, 50000).astype(int)

    # Vehicle value: $5,000 - $100,000
    df["vehicle_value"] = np.clip((df["vehicle_value"] * 25000) + 30000, 5000, 100000).astype(int)

    # Deductible: $250 - $2,500
    df["deductible_amount"] = np.clip((df["deductible_amount"] * 550) + 750, 250, 2500).astype(int)

    # Coverage limit: $50,000 - $500,000
    df["coverage_limit"] = np.clip((df["coverage_limit"] * 112500) + 150000, 50000, 500000).astype(int)

    # Years insured: 0-50 years
    df["years_insured"] = np.clip((df["years_insured"] * 12.5) + 5, 0, 50).astype(int)

    # Previous claims: 0-5 claims
    df["previous_claims"] = np.clip((df["previous_claims"] * 1.25), 0, 5).astype(int)

    # Credit score: 300-850
    df["credit_score"] = np.clip((df["credit_score"] * 137.5) + 425, 300, 850).astype(int)

    # Convert categorical features to appropriate types
    df["marital_status"] = np.random.choice([0, 1], size=len(df))  # 0=Single, 1=Married
    df["occupation_risk"] = np.clip(df["occupation_risk"] * 2, 0, 3).astype(int)  # 0-3 risk levels
    df["location_risk_score"] = np.clip(df["location_risk_score"] * 25, 0, 100).astype(int)  # 0-100 risk score
    df["vehicle_age"] = np.clip((df["vehicle_age"] * 5) + 2, 0, 15).astype(int)  # 0-15 years old
    df["policy_type"] = np.random.choice([0, 1, 2], size=len(df))  # 0=Basic, 1=Standard, 2=Premium
    df["discount_eligibility"] = np.random.choice([0, 1], size=len(df))  # 0=No, 1=Yes
    df["safe_driver_discount"] = np.random.choice([0, 1], size=len(df))  # 0=No, 1=Yes

    return df


def _add_demographic_features(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add demographic features for fairness analysis using safe utilities."""
    from ..data.gen_utils import safe_categorical_select  # noqa: PLC0415

    np.random.seed(random_state)

    # Gender (protected attribute)
    df["gender"] = np.random.choice(["Male", "Female"], size=len(df))

    # Age groups (protected attribute) - using safe categorical selection
    conditions = [
        (df["age"] < 25),
        (df["age"] >= 25) & (df["age"] < 35),
        (df["age"] >= 35) & (df["age"] < 50),
        (df["age"] >= 50) & (df["age"] < 65),
        (df["age"] >= 65),
    ]
    choices = ["Young", "Young_Adult", "Middle_Age", "Senior", "Elderly"]
    df["age_group"] = safe_categorical_select(conditions, choices, "Unknown")

    # Income brackets (socioeconomic status) - using safe categorical selection
    conditions = [
        (df["annual_mileage"] < 8000),
        (df["annual_mileage"] >= 8000) & (df["annual_mileage"] < 15000),
        (df["annual_mileage"] >= 15000),
    ]
    choices = ["Low", "Middle", "High"]
    df["income_bracket"] = safe_categorical_select(conditions, choices, "Unknown")

    return df


def create_insurance_risk_config() -> dict[str, Any]:
    """Create example configuration for insurance risk assessment audit.

    Returns:
        Configuration dictionary for YAML serialization

    """
    return {
        "audit_profile": "insurance_risk_assessment",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "~/.glassalpha/data/insurance_risk.csv",
            "target_column": "claim_outcome",
            "protected_attributes": ["gender", "age_group"],
        },
        "model": {
            "type": "lightgbm",
            "params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
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
    df = generate_insurance_risk_dataset(
        n_samples=10000,
        random_state=42,
        output_path="~/.glassalpha/data/insurance_risk.csv",
    )

    print("Insurance Risk Dataset Generated:")
    print(f"Shape: {df.shape}")
    print(f"Claim rate: {df['claim_outcome'].mean():.1%}")
    print(f"Protected groups: {df['gender'].value_counts().to_dict()}")
    print(f"Age groups: {df['age_group'].value_counts().to_dict()}")
