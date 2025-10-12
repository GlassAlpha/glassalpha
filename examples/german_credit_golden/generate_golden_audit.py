#!/usr/bin/env python3
"""Generate golden audit package for German Credit dataset.

This script creates a complete audit package including:
- Audit report (HTML)
- Manifest JSON with full lineage
- Config YAML used
- SHA-256 checksums for verification

Usage: python3 generate_golden_audit.py
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import xgboost as xgb

try:
    # Load German Credit data
    from glassalpha.datasets import load_german_credit

    print("Loading German Credit dataset...")
    data = load_german_credit(train_test_split=True, test_size=0.3, random_state=42)
    X_train, X_test = data

    # Simple preprocessing for golden example
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Clean column names for XGBoost compatibility
    X_train.columns = [
        col.replace("[", "").replace("]", "").replace("<", "").replace(">", "") for col in X_train.columns
    ]
    X_test.columns = [col.replace("[", "").replace("]", "").replace("<", "").replace(">", "") for col in X_test.columns]

    # Align columns between train and test
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Create target variable (dummy for demonstration)
    y_train = (X_train.sum(axis=1) > X_train.sum(axis=1).median()).astype(int)
    y_test = (X_test.sum(axis=1) > X_test.sum(axis=1).median()).astype(int)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Train a simple XGBoost model for demonstration
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,  # Suppress XGBoost output
    )

    model.fit(X_train, y_train)

    # Test the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model accuracy - Train: {train_score:.3f}, Test: {test_score:.3f}")

    # Now run the audit
    print("Running audit...")
    from glassalpha.api.audit import from_model
    from glassalpha.config import load_config

    # Load config for audit
    config = load_config("config.yaml")

    result = from_model(
        model,
        X_test,
        y_test,
        random_seed=config.reproducibility.random_seed,
    )

    print("✅ Audit completed successfully!")
    print(
        f"📊 Generated performance metrics: {len(result.performance.__dict__) if hasattr(result.performance, '__dict__') else 'available'}"
    )
    print(
        f"📈 Generated explanations: {len(result.explanations.__dict__) if hasattr(result.explanations, '__dict__') else 'available'}"
    )
    print(
        f"📋 Generated fairness analysis: {len(result.fairness.__dict__) if hasattr(result.fairness, '__dict__') else 'available'}"
    )

    # Generate the HTML report
    print("Generating HTML report...")
    html_path = Path("audit_report.html")
    result.to_html(html_path, overwrite=True)
    print(f"📄 HTML report saved to: {html_path}")

    # Generate the manifest
    print("Generating manifest...")
    import json

    manifest = result.manifest
    manifest_path = Path("manifest.json")
    manifest_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    print(f"📋 Manifest saved to: {manifest_path}")

    # Save the config for reference
    config_path = Path("config.yaml")
    print(f"📋 Config file: {config_path}")

    # Generate checksums
    print("Generating checksums...")
    import hashlib

    checksums = []
    for file_path in [html_path, manifest_path, config_path]:
        if file_path.exists():
            with open(file_path, "rb") as f:
                content = f.read()
                sha256 = hashlib.sha256(content).hexdigest()
                checksums.append(f"{sha256}  {file_path.name}")

    checksums_path = Path("SHA256SUMS.txt")
    checksums_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"🔐 Checksums saved to: {checksums_path}")

    print("\n🎉 Golden audit package created successfully!")
    print("Files generated:")
    for file_path in [html_path, manifest_path, config_path, checksums_path]:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  - {file_path.name} ({size:,} bytes)")

    print("\nTo verify reproducibility:")
    print("1. Save this directory as a zip file")
    print("2. Share the zip file with auditors")
    print("3. Auditors can run: python3 -m glassalpha verify-evidence-pack audit_package.zip")

except Exception as e:
    print(f"❌ Error generating golden audit: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
