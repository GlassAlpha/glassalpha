"""Customer Segmentation Dataset Generator.

This module generates synthetic customer data for demonstrating multi-class
classification in marketing segmentation. The dataset simulates customer
behavior patterns for targeted marketing compliance auditing.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_customer_segmentation_dataset(
    n_samples: int = 20000,
    n_classes: int = 4,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic customer segmentation dataset.

    Args:
        n_samples: Number of customers to generate
        n_classes: Number of customer segments (default 4)
        random_state: Random seed for reproducibility
        output_path: Optional path to save the dataset

    Returns:
        DataFrame with customer data and segment labels

    """
    np.random.seed(random_state)

    # Generate base features using sklearn for multi-class
    X, y = make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=1,
        n_classes=n_classes,
        class_sep=1.0,
        random_state=random_state,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "age",
        "income",
        "education_years",
        "household_size",
        "children_count",
        "marital_status",
        "home_ownership",
        "employment_years",
        "occupation_type",
        "credit_score",
        "savings_amount",
        "investment_portfolio",
        "purchase_frequency",
        "avg_order_value",
        "category_preferences",
        "loyalty_program_member",
        "social_media_engagement",
        "mobile_app_usage",
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform features to realistic customer data ranges
    df = _transform_customer_features(df)
    df["customer_segment"] = y  # Multi-class: 0-3 customer segments

    # Add demographic features for fairness analysis
    df = _add_customer_demographics(df, random_state)

    # Add customer metadata
    df["customer_id"] = [f"CUST_{i:08d}" for i in range(len(df))]
    df["registration_date"] = pd.date_range(
        start="2020-01-01",
        periods=len(df),
        freq="D",
    ).strftime("%Y-%m-%d")

    # Reorder columns
    cols = ["customer_id", "registration_date"] + feature_names + ["customer_segment"]
    df = df[cols]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved customer segmentation dataset to {output_path}")

    return df


def _transform_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform generated features to realistic customer data ranges."""
    # Age: 18-80 years
    df["age"] = np.clip((df["age"] * 15) + 35, 18, 80).astype(int)

    # Income: $20,000 - $200,000
    df["income"] = np.clip((df["income"] * 45000) + 50000, 20000, 200000).astype(int)

    # Education: 8-25 years (grade school through graduate)
    df["education_years"] = np.clip((df["education_years"] * 4) + 12, 8, 25).astype(int)

    # Household size: 1-8 people
    df["household_size"] = np.clip((df["household_size"] * 2) + 2, 1, 8).astype(int)

    # Children count: 0-5 children
    df["children_count"] = np.clip((df["children_count"] * 1.25), 0, 5).astype(int)

    # Marital status: 0=single, 1=married, 2=divorced, 3=widowed
    df["marital_status"] = np.clip(df["marital_status"] * 1, 0, 3).astype(int)

    # Home ownership: 0=rent, 1=own, 2=mortgage
    df["home_ownership"] = np.clip(df["home_ownership"] * 0.75, 0, 2).astype(int)

    # Employment years: 0-50 years
    df["employment_years"] = np.clip((df["employment_years"] * 12.5) + 5, 0, 50).astype(int)

    # Occupation type: 0=blue collar, 1=white collar, 2=professional, 3=retired
    df["occupation_type"] = np.clip(df["occupation_type"] * 1, 0, 3).astype(int)

    # Credit score: 300-850
    df["credit_score"] = np.clip((df["credit_score"] * 137.5) + 425, 300, 850).astype(int)

    # Savings amount: $0 - $500,000
    df["savings_amount"] = np.clip((df["savings_amount"] * 125000) + 10000, 0, 500000).astype(int)

    # Investment portfolio: $0 - $1,000,000
    df["investment_portfolio"] = np.clip((df["investment_portfolio"] * 250000) + 25000, 0, 1000000).astype(int)

    # Purchase frequency: 0-100 purchases per year
    df["purchase_frequency"] = np.clip((df["purchase_frequency"] * 25) + 5, 0, 100).astype(int)

    # Average order value: $10 - $1,000
    df["avg_order_value"] = np.clip((df["avg_order_value"] * 245) + 50, 10, 1000).round(2)

    # Category preferences: 0-10 (0=no preference, 10=strong preference)
    df["category_preferences"] = np.clip(df["category_preferences"] * 2.5, 0, 10).astype(int)

    # Loyalty program: 0=not member, 1=member
    df["loyalty_program_member"] = np.clip(df["loyalty_program_member"] * 0.25, 0, 1).astype(int)

    # Social media engagement: 0-100 (engagement score)
    df["social_media_engagement"] = np.clip(df["social_media_engagement"] * 25, 0, 100).astype(int)

    # Mobile app usage: 0-100 (app usage score)
    df["mobile_app_usage"] = np.clip(df["mobile_app_usage"] * 25, 0, 100).astype(int)

    return df


def _add_customer_demographics(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add demographic features for fairness analysis in customer segmentation."""
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

    # Income brackets (socioeconomic status)
    conditions = [
        (df["income"] < 40000),
        (df["income"] >= 40000) & (df["income"] < 80000),
        (df["income"] >= 80000) & (df["income"] < 150000),
        (df["income"] >= 150000),
    ]
    choices = ["Low", "Lower_Middle", "Upper_Middle", "High"]
    df["income_bracket"] = np.select(conditions, choices, default="Unknown")

    # Education level proxy
    conditions = [
        (df["education_years"] < 12),
        (df["education_years"] >= 12) & (df["education_years"] < 16),
        (df["education_years"] >= 16) & (df["education_years"] < 20),
        (df["education_years"] >= 20),
    ]
    choices = ["High_School", "Bachelor", "Master", "Doctorate"]
    df["education_level"] = np.select(conditions, choices, default="Unknown")

    return df


def create_customer_segmentation_config() -> dict[str, Any]:
    """Create example configuration for customer segmentation audit.

    Returns:
        Configuration dictionary for YAML serialization

    """
    return {
        "audit_profile": "customer_segmentation",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "~/.glassalpha/data/customer_segmentation.csv",
            "target_column": "customer_segment",
            "protected_attributes": ["gender", "age_group"],
        },
        "model": {
            "type": "sklearn_generic",
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
        },
        "explainers": {
            "strategy": "first_compatible",
            "priority": ["kernelshap"],  # Random Forest needs KernelSHAP
            "config": {
                "kernelshap": {
                    "n_samples": 1000,
                    "background_size": 100,
                },
            },
        },
        "metrics": {
            "performance": {
                "metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            },
            "fairness": {
                "metrics": ["demographic_parity", "equal_opportunity"],
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
    df = generate_customer_segmentation_dataset(
        n_samples=20000,
        n_classes=4,
        random_state=42,
        output_path="~/.glassalpha/data/customer_segmentation.csv",
    )

    print("Customer Segmentation Dataset Generated:")
    print(f"Shape: {df.shape}")
    print(f"Segment distribution: {df['customer_segment'].value_counts().to_dict()}")
    print(f"Protected groups: {df['gender'].value_counts().to_dict()}")
    print(f"Age groups: {df['age_group'].value_counts().to_dict()}")
    print(f"Income brackets: {df['income_bracket'].value_counts().to_dict()}")
