"""CCPA Compliance Dataset Generator.

This module generates synthetic California consumer data for demonstrating ML model auditing
in CCPA compliance scenarios. The dataset simulates consumer profiling for
marketing with California Consumer Privacy Act requirements.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_ccpa_compliance_dataset(
    n_samples: int = 8000,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic CCPA compliance dataset.

    Args:
        n_samples: Number of California consumer records to generate
        random_state: Random seed for reproducibility
        output_path: Optional path to save the dataset

    Returns:
        DataFrame with California consumer data and privacy decisions

    """
    np.random.seed(random_state)

    # Generate base features using sklearn
    X, y = make_classification(
        n_samples=n_samples,
        n_features=14,
        n_informative=9,
        n_redundant=3,
        n_clusters_per_class=1,
        class_sep=0.85,
        random_state=random_state,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "age",
        "income_usd",
        "education_years",
        "household_size",
        "marketing_consent_given",
        "data_sharing_opt_out",
        "tracking_cookies_accepted",
        "location_services_enabled",
        "social_media_activity",
        "mobile_app_usage",
        "online_purchase_frequency",
        "subscription_services",
        "loyalty_program_member",
        "california_resident",
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform features to realistic California consumer data ranges
    df = _transform_ccpa_features(df)
    df["automated_decision_consent"] = y  # Binary: 1 = consent for automated decisions, 0 = opt-out

    # Add demographic features for fairness analysis
    df = _add_ccpa_demographics(df, random_state)

    # Add CCPA-specific metadata
    df["consumer_id"] = [f"CA_CONS_{i:08d}" for i in range(len(df))]
    df["data_collection_date"] = pd.date_range(
        start="2023-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")
    df["last_consent_update"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")

    # Add CCPA consumer rights indicators
    df["data_portability_requested"] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
    df["data_deletion_requested"] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
    df["do_not_sell_requested"] = np.random.choice([0, 1], size=len(df), p=[0.90, 0.10])
    df["profiling_category"] = np.random.choice(
        [
            "marketing",
            "credit_decisions",
            "employment",
            "insurance",
        ],
        size=len(df),
    )

    # Reorder columns
    cols = (
        ["consumer_id", "data_collection_date", "last_consent_update"]
        + feature_names
        + [
            "automated_decision_consent",
            "data_portability_requested",
            "data_deletion_requested",
            "do_not_sell_requested",
            "profiling_category",
        ]
    )
    df = df[cols]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CCPA compliance dataset to {output_path}")

    return df


def _transform_ccpa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform generated features to realistic California consumer data ranges."""
    # Age: 18-85 years
    df["age"] = np.clip((df["age"] * 17) + 30, 18, 85).astype(int)

    # Income: $25,000 - $200,000
    df["income_usd"] = np.clip((df["income_usd"] * 43750) + 50000, 25000, 200000).astype(int)

    # Education: 8-25 years
    df["education_years"] = np.clip((df["education_years"] * 4) + 12, 8, 25).astype(int)

    # Household size: 1-6 people
    df["household_size"] = np.clip((df["household_size"] * 1.5) + 2, 1, 6).astype(int)

    # Privacy preferences
    df["marketing_consent_given"] = np.clip(df["marketing_consent_given"] * 0.25, 0, 1).astype(int)
    df["data_sharing_opt_out"] = np.clip(df["data_sharing_opt_out"] * 0.15, 0, 1).astype(int)
    df["tracking_cookies_accepted"] = np.clip(df["tracking_cookies_accepted"] * 0.3, 0, 1).astype(int)
    df["location_services_enabled"] = np.clip(df["location_services_enabled"] * 0.2, 0, 1).astype(int)

    # Digital behavior
    df["social_media_activity"] = np.clip(df["social_media_activity"] * 25, 0, 100).astype(int)
    df["mobile_app_usage"] = np.clip((df["mobile_app_usage"] * 50) + 10, 0, 200).astype(int)

    # Purchase behavior
    df["online_purchase_frequency"] = np.clip((df["online_purchase_frequency"] * 15) + 3, 0, 60).astype(int)
    df["subscription_services"] = np.clip((df["subscription_services"] * 4), 0, 12).astype(int)

    # Loyalty program: 0-3 tiers
    df["loyalty_program_member"] = np.clip(df["loyalty_program_member"] * 0.75, 0, 3).astype(int)

    # California residency: 0=no, 1=yes
    df["california_resident"] = np.clip(df["california_resident"] * 0.25, 0, 1).astype(int)

    return df


def _add_ccpa_demographics(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add demographic features for CCPA fairness analysis."""
    np.random.seed(random_state)

    # Gender (protected attribute)
    df["gender"] = np.random.choice(["Male", "Female"], size=len(df))

    # Age groups (protected attribute)
    conditions = [
        (df["age"] < 25),
        (df["age"] >= 25) & (df["age"] < 35),
        (df["age"] >= 35) & (df["age"] < 50),
        (df["age"] >= 50) & (df["age"] < 65),
        (df["age"] >= 65),
    ]
    choices = ["Young", "Young_Adult", "Middle_Age", "Senior", "Elderly"]
    df["age_group"] = np.select(conditions, choices, default="Unknown")

    # California regions (for geographic analysis)
    df["california_region"] = np.random.choice(
        [
            "Los_Angeles",
            "San_Francisco",
            "San_Diego",
            "Sacramento",
            "Other_CA",
        ],
        size=len(df),
        p=[0.35, 0.25, 0.15, 0.10, 0.15],
    )

    # Income brackets (socioeconomic analysis)
    conditions = [
        (df["income_usd"] < 50000),
        (df["income_usd"] >= 50000) & (df["income_usd"] < 100000),
        (df["income_usd"] >= 100000) & (df["income_usd"] < 150000),
        (df["income_usd"] >= 150000),
    ]
    choices = ["Low", "Middle", "Upper_Middle", "High"]
    df["income_bracket"] = np.select(conditions, choices, default="Unknown")

    return df


def create_ccpa_compliance_config() -> dict[str, Any]:
    """Create example configuration for CCPA compliance audit.

    Returns:
        Configuration dictionary for YAML serialization

    """
    return {
        "audit_profile": "ccpa_compliance",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "~/.glassalpha/data/ccpa_compliance.csv",
            "target_column": "automated_decision_consent",
            "protected_attributes": ["gender", "age_group", "income_bracket"],
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
                "metrics": ["demographic_parity", "equal_opportunity", "predictive_parity"],
                "config": {
                    "demographic_parity": {"threshold": 0.05},
                    "equal_opportunity": {"threshold": 0.05},
                    "predictive_parity": {"threshold": 0.05},
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
    df = generate_ccpa_compliance_dataset(
        n_samples=8000,
        random_state=42,
        output_path="~/.glassalpha/data/ccpa_compliance.csv",
    )

    print("CCPA Compliance Dataset Generated:")
    print(f"Shape: {df.shape}")
    print(f"Automated decision consent rate: {df['automated_decision_consent'].mean():.1%}")
    print(f"Protected groups: {df['gender'].value_counts().to_dict()}")
    print(f"Age groups: {df['age_group'].value_counts().to_dict()}")
    print(f"California regions: {df['california_region'].value_counts().to_dict()}")
    print(f"Consumer rights exercised: {df['do_not_sell_requested'].mean():.1%} opt-outs")
