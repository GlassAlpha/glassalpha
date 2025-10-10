"""Financial Fraud Detection Dataset Generator.

This module generates synthetic financial transaction data for demonstrating
ML model auditing in fraud detection systems. The dataset simulates credit
card transactions with fraud indicators for regulatory compliance testing.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_fraud_detection_dataset(
    n_samples: int = 50000,
    fraud_rate: float = 0.01,  # 1% fraud rate (realistic)
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic fraud detection dataset.

    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Fraction of transactions that are fraudulent
        random_state: Random seed for reproducibility
        output_path: Optional path to save the dataset

    Returns:
        DataFrame with transaction data and fraud labels

    """
    np.random.seed(random_state)

    # Generate base features using sklearn
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=1.2,  # Higher separation for fraud detection
        weights=[1 - fraud_rate, fraud_rate],  # Imbalanced classes
        random_state=random_state,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "amount",
        "merchant_category",
        "transaction_hour",
        "transaction_day",
        "cardholder_age",
        "account_age_months",
        "transaction_count_24h",
        "amount_avg_24h",
        "amount_max_24h",
        "location_distance",
        "device_fingerprint_risk",
        "ip_geolocation_risk",
        "card_type",
        "transaction_type",
        "merchant_country_risk",
        "velocity_check_failed",
        "amount_deviation_score",
        "time_since_last_txn",
        "weekend_transaction",
        "international_transaction",
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform features to realistic transaction data ranges
    df = _transform_transaction_features(df)
    df["is_fraud"] = y  # Binary: 1 = fraud, 0 = legitimate

    # Add demographic features for fairness analysis
    df = _add_fraud_demographics(df, random_state)

    # Add transaction metadata
    df["transaction_id"] = [f"TXN_{i:08d}" for i in range(len(df))]
    df["timestamp"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="1min",  # One transaction per minute
    )

    # Reorder columns
    cols = ["transaction_id", "timestamp"] + feature_names + ["is_fraud"]
    df = df[cols]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved fraud detection dataset to {output_path}")

    return df


def _transform_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform generated features to realistic transaction data ranges."""
    # Transaction amount: $1 - $10,000 (log-normal distribution for transactions)
    df["amount"] = np.exp((df["amount"] * 2) + 2).clip(1, 10000).round(2)

    # Merchant category: 0-15 categories
    df["merchant_category"] = np.clip(df["merchant_category"] * 4, 0, 15).astype(int)

    # Transaction hour: 0-23
    df["transaction_hour"] = np.clip((df["transaction_hour"] * 6) + 12, 0, 23).astype(int)

    # Transaction day: 0-6 (0=Monday, 6=Sunday)
    df["transaction_day"] = np.clip(df["transaction_day"] * 1.7, 0, 6).astype(int)

    # Cardholder age: 18-85
    df["cardholder_age"] = np.clip((df["cardholder_age"] * 17) + 30, 18, 85).astype(int)

    # Account age: 0-120 months
    df["account_age_months"] = np.clip((df["account_age_months"] * 30) + 12, 0, 120).astype(int)

    # Transaction count in last 24h: 0-50
    df["transaction_count_24h"] = np.clip((df["transaction_count_24h"] * 12.5), 0, 50).astype(int)

    # Average amount in last 24h: $0 - $5,000
    df["amount_avg_24h"] = np.clip((df["amount_avg_24h"] * 1250) + 50, 0, 5000).round(2)

    # Max amount in last 24h: $0 - $15,000
    df["amount_max_24h"] = np.clip((df["amount_max_24h"] * 3750) + 100, 0, 15000).round(2)

    # Location distance: 0-1000 miles
    df["location_distance"] = np.clip((df["location_distance"] * 250) + 5, 0, 1000).round(1)

    # Device fingerprint risk: 0-100 risk score
    df["device_fingerprint_risk"] = np.clip(df["device_fingerprint_risk"] * 25, 0, 100).astype(int)

    # IP geolocation risk: 0-100 risk score
    df["ip_geolocation_risk"] = np.clip(df["ip_geolocation_risk"] * 25, 0, 100).astype(int)

    # Card type: 0=debit, 1=credit, 2=prepaid
    df["card_type"] = np.clip(df["card_type"] * 0.75, 0, 2).astype(int)

    # Transaction type: 0=online, 1=in-store, 2=ATM
    df["transaction_type"] = np.clip(df["transaction_type"] * 0.75, 0, 2).astype(int)

    # Merchant country risk: 0-100 risk score
    df["merchant_country_risk"] = np.clip(df["merchant_country_risk"] * 25, 0, 100).astype(int)

    # Velocity check failed: 0=no, 1=yes
    df["velocity_check_failed"] = np.clip(df["velocity_check_failed"] * 0.25, 0, 1).astype(int)

    # Amount deviation score: 0-100 (how unusual the amount is)
    df["amount_deviation_score"] = np.clip(df["amount_deviation_score"] * 25, 0, 100).astype(int)

    # Time since last transaction: 0-1440 minutes (0-24 hours)
    df["time_since_last_txn"] = np.clip((df["time_since_last_txn"] * 360) + 30, 0, 1440).astype(int)

    # Weekend transaction: 0=weekday, 1=weekend
    df["weekend_transaction"] = np.clip(df["weekend_transaction"] * 0.25, 0, 1).astype(int)

    # International transaction: 0=domestic, 1=international
    df["international_transaction"] = np.clip(df["international_transaction"] * 0.1, 0, 1).astype(int)

    return df


def _add_fraud_demographics(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add demographic features for fairness analysis in fraud detection."""
    np.random.seed(random_state)

    # Gender (protected attribute)
    df["gender"] = np.random.choice(["Male", "Female"], size=len(df))

    # Age groups (protected attribute)
    conditions = [
        (df["cardholder_age"] < 25),
        (df["cardholder_age"] >= 25) & (df["cardholder_age"] < 35),
        (df["cardholder_age"] >= 35) & (df["cardholder_age"] < 50),
        (df["cardholder_age"] >= 50) & (df["cardholder_age"] < 65),
        (df["cardholder_age"] >= 65),
    ]
    choices = ["Young", "Young_Adult", "Middle_Age", "Senior", "Elderly"]
    df["age_group"] = np.select(conditions, choices, default="Unknown")

    # Income proxy (based on transaction patterns)
    conditions = [
        (df["amount_avg_24h"] < 50),
        (df["amount_avg_24h"] >= 50) & (df["amount_avg_24h"] < 200),
        (df["amount_avg_24h"] >= 200),
    ]
    choices = ["Low", "Middle", "High"]
    df["income_bracket"] = np.select(conditions, choices, default="Unknown")

    return df


def create_fraud_detection_config() -> dict[str, Any]:
    """Create example configuration for fraud detection audit.

    Returns:
        Configuration dictionary for YAML serialization

    """
    return {
        "audit_profile": "fraud_detection",
        "reproducibility": {"random_seed": 42},
        "data": {
            "path": "~/.glassalpha/data/fraud_detection.csv",
            "target_column": "is_fraud",
            "protected_attributes": ["gender", "age_group"],
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
                "scale_pos_weight": 99,  # Handle class imbalance
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
                    "demographic_parity": {"threshold": 0.02},  # Stricter for fraud
                    "equal_opportunity": {"threshold": 0.02},
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
    df = generate_fraud_detection_dataset(
        n_samples=50000,
        fraud_rate=0.01,  # 1% fraud rate
        random_state=42,
        output_path="~/.glassalpha/data/fraud_detection.csv",
    )

    print("Fraud Detection Dataset Generated:")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    print(f"Protected groups: {df['gender'].value_counts().to_dict()}")
    print(f"Age groups: {df['age_group'].value_counts().to_dict()}")
    print(f"Transaction amount range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
