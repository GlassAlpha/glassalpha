"""GDPR Compliance Dataset Generator.

This module generates synthetic customer data for demonstrating ML model auditing
in GDPR compliance scenarios. The dataset simulates customer profiling for
marketing with EU data protection regulation requirements.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_gdpr_compliance_dataset(
    n_samples: int = 12000,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic GDPR compliance dataset.

    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility
        output_path: Optional path to save the dataset

    Returns:
        DataFrame with customer data and marketing consent decisions

    """
    np.random.seed(random_state)

    # Generate base features using sklearn
    X, y = make_classification(
        n_samples=n_samples,
        n_features=16,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=random_state,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "age",
        "income_eur",
        "education_years",
        "household_size",
        "marketing_emails_opened",
        "website_visits_monthly",
        "purchase_frequency",
        "avg_order_value_eur",
        "customer_since_years",
        "social_media_engagement",
        "mobile_app_sessions",
        "newsletter_subscriptions",
        "loyalty_program_tier",
        "geographic_region",
        "language_preference",
        "data_processing_consent",
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform features to realistic EU customer data ranges
    df = _transform_gdpr_features(df)
    df["marketing_consent_granted"] = y  # Binary: 1 = consent granted, 0 = denied

    # Add demographic features for fairness analysis
    df = _add_gdpr_demographics(df, random_state)

    # Add GDPR-specific metadata
    df["customer_id"] = [f"EU_CUST_{i:08d}" for i in range(len(df))]
    df["consent_date"] = pd.date_range(
        start="2023-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")
    df["last_data_processing"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")

    # Add GDPR Article 22 indicators
    df["automated_decision_subject"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    df["profiling_category"] = np.random.choice(
        [
            "marketing",
            "credit_scoring",
            "insurance_risk",
            "employment_screening",
        ],
        size=len(df),
    )

    # Reorder columns
    cols = (
        ["customer_id", "consent_date", "last_data_processing"]
        + feature_names
        + [
            "marketing_consent_granted",
            "automated_decision_subject",
            "profiling_category",
        ]
    )
    df = df[cols]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved GDPR compliance dataset to {output_path}")

    return df


def _transform_gdpr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform generated features to realistic EU customer data ranges."""
    # Age: 18-80 years
    df["age"] = np.clip((df["age"] * 15) + 35, 18, 80).astype(int)

    # Income: €15,000 - €150,000
    df["income_eur"] = np.clip((df["income_eur"] * 33750) + 40000, 15000, 150000).astype(int)

    # Education: 8-25 years
    df["education_years"] = np.clip((df["education_years"] * 4) + 12, 8, 25).astype(int)

    # Household size: 1-6 people
    df["household_size"] = np.clip((df["household_size"] * 1.5) + 2, 1, 6).astype(int)

    # Marketing engagement
    df["marketing_emails_opened"] = np.clip((df["marketing_emails_opened"] * 25) + 5, 0, 100).astype(int)
    df["website_visits_monthly"] = np.clip((df["website_visits_monthly"] * 30) + 5, 0, 150).astype(int)

    # Purchase behavior
    df["purchase_frequency"] = np.clip((df["purchase_frequency"] * 12) + 2, 0, 50).astype(int)
    df["avg_order_value_eur"] = np.clip((df["avg_order_value_eur"] * 75) + 25, 5, 500).round(2)

    # Customer tenure: 0-20 years
    df["customer_since_years"] = np.clip((df["customer_since_years"] * 5) + 2, 0, 20).astype(int)

    # Digital engagement
    df["social_media_engagement"] = np.clip(df["social_media_engagement"] * 25, 0, 100).astype(int)
    df["mobile_app_sessions"] = np.clip((df["mobile_app_sessions"] * 50) + 10, 0, 200).astype(int)
    df["newsletter_subscriptions"] = np.clip((df["newsletter_subscriptions"] * 3), 0, 8).astype(int)

    # Loyalty program: 0-4 tiers
    df["loyalty_program_tier"] = np.clip(df["loyalty_program_tier"] * 1, 0, 4).astype(int)

    # Geographic region: 0-4 (EU countries/regions)
    df["geographic_region"] = np.clip(df["geographic_region"] * 1, 0, 4).astype(int)

    # Language preference: 0-4 (major EU languages)
    df["language_preference"] = np.clip(df["language_preference"] * 1, 0, 4).astype(int)

    # Data processing consent: 0=no, 1=yes
    df["data_processing_consent"] = np.clip(df["data_processing_consent"] * 0.25, 0, 1).astype(int)

    return df


def _add_gdpr_demographics(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add demographic features for GDPR fairness analysis."""
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

    # EU citizenship status (GDPR protected)
    df["eu_citizenship"] = np.random.choice(["EU_Citizen", "Non_EU_Resident"], size=len(df), p=[0.85, 0.15])

    # Data protection authority region
    df["dpa_region"] = np.random.choice(
        [
            "Germany",
            "France",
            "Italy",
            "Spain",
            "Netherlands",
            "Other_EU",
        ],
        size=len(df),
        p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
    )

    return df


def create_gdpr_compliance_config() -> dict[str, Any]:
    """Create example configuration for GDPR compliance audit.

    Returns:
        Configuration dictionary for YAML serialization

    """
    return {
        "audit_profile": "gdpr_compliance",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "~/.glassalpha/data/gdpr_compliance.csv",
            "target_column": "marketing_consent_granted",
            "protected_attributes": ["gender", "age_group", "eu_citizenship"],
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
    df = generate_gdpr_compliance_dataset(
        n_samples=12000,
        random_state=42,
        output_path="~/.glassalpha/data/gdpr_compliance.csv",
    )

    print("GDPR Compliance Dataset Generated:")
    print(f"Shape: {df.shape}")
    print(f"Consent rate: {df['marketing_consent_granted'].mean():.1%}")
    print(f"Protected groups: {df['gender'].value_counts().to_dict()}")
    print(f"Age groups: {df['age_group'].value_counts().to_dict()}")
    print(f"EU citizenship: {df['eu_citizenship'].value_counts().to_dict()}")
    print(f"Automated decisions: {df['automated_decision_subject'].mean():.1%} of customers")
