"""Explanation commands: reasons and recourse.

ARCHITECTURE NOTE: These are separate from audit command intentionally:
- Different data format requirements (preprocessed vs raw)
- Different regulatory purposes (ECOA notices vs audit reports)
- Different workflows (post-audit analysis vs initial generation)
"""

import logging
from pathlib import Path

import typer

from glassalpha.cli.exit_codes import ExitCode

logger = logging.getLogger(__name__)


def reasons(  # pragma: no cover - CLI command
    # ARCHITECTURE NOTE: Separate command from `audit` is intentional
    # - Different data format requirement (preprocessed vs raw)
    # - Different regulatory purpose (ECOA adverse action notices)
    # - Different workflow (post-audit analysis vs initial audit generation)
    # DO NOT merge into audit command - this serves distinct compliance workflow
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl",
        exists=True,
        file_okay=True,
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to test data file (CSV). MUST be preprocessed data (e.g., models/test_data.csv from audit).",
        file_okay=True,
    ),
    instance: int = typer.Option(
        ...,
        "--instance",
        "-i",
        help="Row index of instance to explain (0-based)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to reason codes configuration YAML",
        file_okay=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for output notice file (defaults to stdout)",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Decision threshold for approved/denied",
    ),
    top_n: int = typer.Option(
        4,
        "--top-n",
        "-n",
        help="Number of reason codes to generate (ECOA typical: 4)",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: 'text' or 'json'",
    ),
):
    """Generate ECOA-compliant reason codes for adverse action notice.

    This command extracts top-N negative feature contributions from a trained model
    to explain why a specific instance was denied (or approved). Output is formatted
    as an ECOA-compliant adverse action notice.

    Requirements:
        - Trained model with SHAP-compatible architecture
        - Test dataset with same features as training
        - Instance index to explain

    Examples:
        # Generate reason codes for instance 42
        glassalpha reasons \\
            --model models/german_credit.pkl \\
            --data data/test.csv \\
            --instance 42 \\
            --output notices/instance_42.txt

        # With custom config
        glassalpha reasons -m model.pkl -d test.csv -i 10 -c config.yaml

        # JSON output
        glassalpha reasons -m model.pkl -d test.csv -i 5 --format json

        # Custom threshold and top-N
        glassalpha reasons -m model.pkl -d test.csv -i 0 --threshold 0.6 --top-n 3

    """
    # Lazy imports - only when function is actually called
    import json

    import joblib

    try:
        import pandas as pd  # Lazy import for performance

        # Load configuration if provided
        protected_attributes = None
        organization = "[Organization Name]"
        contact_info = "[Contact Information]"
        seed = 42

        if config and config.exists():
            from glassalpha.config import load_config

            cfg = load_config(config)
            protected_attributes = getattr(cfg.data, "protected_attributes", None) if hasattr(cfg, "data") else None
            seed = getattr(cfg.reproducibility, "random_seed", 42) if hasattr(cfg, "reproducibility") else 42
            # Load organization info from config if available
            if hasattr(cfg, "reason_codes"):
                organization = getattr(cfg.reason_codes, "organization", organization)
                contact_info = getattr(cfg.reason_codes, "contact_info", contact_info)

        # Check if data file exists, look for model-compatible test data first
        if not data.exists():
            # Smart path detection: Check if model directory has test_data.csv
            model_dir = model.parent
            model_test_data = model_dir / "test_data.csv"

            if model_test_data.exists():
                typer.echo(f"💡 Using test data from model directory: {model_test_data}")
                typer.echo("   (This data matches the model's expected feature structure)")
                typer.echo()
                data = model_test_data
            else:
                # Data file not found - provide clear error with solutions
                typer.secho(f"❌ Error: Test data file not found: {data}", fg=typer.colors.RED, err=True)
                typer.echo()
                typer.echo("The reason codes/recourse commands require test data that matches")
                typer.echo("the model's expected feature structure (including preprocessing).")
                typer.echo()
                typer.echo("📋 Solutions:")
                typer.echo()
                typer.echo("  1. Use test data saved during audit generation:")
                typer.echo(f"     --data {model_test_data}")
                typer.echo()
                typer.echo("  2. Run audit to generate and save test data:")
                typer.echo("     glassalpha audit --config config.yaml")
                typer.echo("     (Automatically saves to models/test_data.csv)")
                typer.echo()
                typer.echo("  3. Provide your own test CSV with same features as training")
                typer.echo("     (after preprocessing, including one-hot encoding)")
                typer.echo()
                typer.echo("📖 For more help:")
                typer.echo("   https://glassalpha.com/guides/reason-codes/")
                typer.echo()
                raise typer.Exit(ExitCode.USER_ERROR)

        typer.echo(f"Loading model from: {model}")
        # Use joblib for loading (matches saving with joblib.dump in audit command)
        loaded = joblib.load(model)

        # Check for companion .meta.json file
        meta_path = Path(model).with_suffix(".meta.json")
        model_metadata = None
        if meta_path.exists():
            try:
                import json

                model_metadata = json.loads(meta_path.read_text())
                typer.echo(f"Found model metadata: {meta_path}")
            except Exception as e:
                logger.warning(f"Failed to load model metadata from {meta_path}: {e}")

        # Handle both old format (model only) and new format (dict with metadata)
        if isinstance(loaded, dict) and "model" in loaded:
            model_obj = loaded["model"]
            expected_features = loaded.get("feature_names")
            preprocessing_info = loaded.get("preprocessing")
        else:
            model_obj = loaded
            expected_features = model_metadata.get("feature_names") if model_metadata else None
            preprocessing_info = None
            if model_metadata and model_metadata.get("preprocessing"):
                preprocessing_dict = model_metadata.get("preprocessing")
                if preprocessing_dict.get("artifact_path"):
                    preprocessing_info = {
                        "mode": "artifact",
                        "artifact_path": preprocessing_dict["artifact_path"],
                        "feature_names_after": expected_features,
                    }

        typer.echo(f"Loading data from: {data}")
        df = pd.read_csv(data)

        # Check if data is already preprocessed by comparing with expected features
        data_is_preprocessed = False
        if preprocessing_info:
            expected_features_list = preprocessing_info.get("feature_names_after", [])
            if expected_features_list:
                # Check column overlap - if >90% match, assume already preprocessed
                matching_cols = len(set(df.columns).intersection(set(expected_features_list)))
                if matching_cols > len(expected_features_list) * 0.9:
                    data_is_preprocessed = True
                    typer.echo("✓ Data is already preprocessed (matches model's expected features)")

        # Apply preprocessing if available and data is NOT already preprocessed
        if preprocessing_info and not data_is_preprocessed:
            typer.echo("Applying preprocessing to match model training...")

            def _apply_preprocessing_from_model_artifact(
                X: "pd.DataFrame",
                preprocessing_info: dict | None,
            ) -> "pd.DataFrame":
                """Apply the same preprocessing that was used during model training."""
                if preprocessing_info is None:
                    return X
                mode = preprocessing_info.get("mode", "auto")

                if mode == "artifact":
                    # Load preprocessing artifact and apply it
                    artifact_path = preprocessing_info.get("artifact_path")
                    if artifact_path:
                        try:
                            import joblib

                            preprocessor = joblib.load(artifact_path)
                            logger.info(f"Applying preprocessing artifact from: {artifact_path}")

                            # Apply preprocessing (assumes target column is not in X)
                            # Use transform, not fit_transform to match training exactly
                            X_transformed = preprocessor.transform(X)

                            # Get expected feature names from preprocessing info or artifact
                            expected_features = preprocessing_info.get("feature_names_after")
                            if expected_features:
                                # Use stored feature names for consistency
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in expected_features
                                ]
                            else:
                                # Fallback: get from artifact
                                feature_names = preprocessor.get_feature_names_out()
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in feature_names
                                ]

                            # Validate feature count matches expectations
                            if len(sanitized_feature_names) != X_transformed.shape[1]:
                                raise ValueError(
                                    f"Feature count mismatch: expected {len(sanitized_feature_names)} "
                                    f"features but got {X_transformed.shape[1]} from preprocessing",
                                )

                            # Return as DataFrame with proper column names
                            return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                        except Exception as e:
                            logger.exception(f"Failed to apply preprocessing artifact: {e}")
                            raise ValueError(
                                f"Could not apply preprocessing artifact: {e}. "
                                "Ensure the preprocessing artifact is accessible and matches the training data.",
                            )
                    else:
                        raise ValueError("Artifact mode specified but no artifact_path provided in model metadata")
                elif mode == "auto":
                    return _apply_auto_preprocessing_from_metadata(X, preprocessing_info)
                else:
                    raise ValueError(f"Unknown preprocessing mode: {mode}")

            def _apply_auto_preprocessing_from_metadata(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                # Check if data is already preprocessed (e.g., loaded from test_data.csv)
                # If X has more columns than expected raw features, it's likely already one-hot encoded
                expected_raw_cols = len(categorical_cols) + len(numeric_cols)
                if len(X.columns) > expected_raw_cols:
                    logger.debug(
                        f"Data appears already preprocessed ({len(X.columns)} cols vs {expected_raw_cols} expected raw). Skipping preprocessing."
                    )
                    return X

                logger.debug(
                    f"Applying auto preprocessing from metadata: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    expected_cols = sorted(categorical_cols + numeric_cols)
                    actual_cols = sorted(X.columns)
                    missing_list = sorted(missing_cols)

                    # Show truncated lists for readability
                    missing_display = ", ".join(missing_list[:10])
                    if len(missing_list) > 10:
                        missing_display += f"... (+{len(missing_list) - 10} more)"

                    expected_display = ", ".join(expected_cols[:10])
                    if len(expected_cols) > 10:
                        expected_display += f"... (+{len(expected_cols) - 10} more)"

                    raise ValueError(
                        f"Test data missing {len(missing_cols)} columns.\n\n"
                        f"Missing columns:\n  {missing_display}\n\n"
                        f"Expected {len(expected_cols)} columns:\n  {expected_display}\n\n"
                        f"Actual {len(actual_cols)} columns:\n  {', '.join(actual_cols)}\n\n"
                        f"Fix options:\n"
                        f"  1. Use original training data (with all features)\n"
                        f"  2. Re-run audit and save test data: glassalpha audit --save-test-data\n"
                        f"  3. Provide config for auto-loading: --config audit_config.yaml"
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                # Use training-time preprocessing info to recreate the same transformation
                # We need to fit on training-like data or use stored parameters
                # For now, fit on the test data but this may not be identical to training
                # TODO: Store and use training preprocessor parameters for perfect reproducibility

                X_transformed = preprocessor.fit_transform(X)
                feature_names = []

                # Reconstruct feature names based on training metadata
                if categorical_cols:
                    cat_transformer = preprocessor.named_transformers_["categorical"]
                    if hasattr(cat_transformer, "get_feature_names_out"):
                        # Use stored feature names if available
                        stored_cat_features = preprocessing_info.get("feature_names_after", [])
                        if stored_cat_features:
                            # Filter for categorical features only
                            feature_names.extend(
                                [
                                    f
                                    for f in stored_cat_features
                                    if any(f.startswith(c + "_") for c in categorical_cols)
                                ],
                            )
                        else:
                            cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                            feature_names.extend(cat_features)
                    else:
                        # Fallback: reconstruct from categories
                        for i, col in enumerate(categorical_cols):
                            unique_vals = cat_transformer.categories_[i]
                            feature_names.extend([f"{col}_{val}" for val in unique_vals])

                if numeric_cols:
                    feature_names.extend(numeric_cols)

                # Validate feature count matches
                if len(feature_names) != X_transformed.shape[1]:
                    logger.warning(
                        f"Feature count mismatch: expected {len(feature_names)} "
                        f"but got {X_transformed.shape[1]} from preprocessing. "
                        "Using generic feature names.",
                    )
                    # Fallback to generic names
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                sanitized_feature_names = [
                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                    for name in feature_names
                ]

                return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)

            def _apply_auto_preprocessing(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                logger.debug(
                    f"Applying auto preprocessing: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    expected_cols = sorted(categorical_cols + numeric_cols)
                    actual_cols = sorted(X.columns)
                    missing_list = sorted(missing_cols)

                    # Show truncated lists for readability
                    missing_display = ", ".join(missing_list[:10])
                    if len(missing_list) > 10:
                        missing_display += f"... (+{len(missing_list) - 10} more)"

                    expected_display = ", ".join(expected_cols[:10])
                    if len(expected_cols) > 10:
                        expected_display += f"... (+{len(expected_cols) - 10} more)"

                    raise ValueError(
                        f"Test data missing {len(missing_cols)} columns.\n\n"
                        f"Missing columns:\n  {missing_display}\n\n"
                        f"Expected {len(expected_cols)} columns:\n  {expected_display}\n\n"
                        f"Actual {len(actual_cols)} columns:\n  {', '.join(actual_cols)}\n\n"
                        f"Fix options:\n"
                        f"  1. Use original training data (with all features)\n"
                        f"  2. Re-run audit and save test data: glassalpha audit --save-test-data\n"
                        f"  3. Provide config for auto-loading: --config audit_config.yaml"
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                try:
                    X_transformed = preprocessor.fit_transform(X)
                    feature_names = []

                    # Reconstruct feature names based on training metadata
                    if categorical_cols:
                        cat_transformer = preprocessor.named_transformers_["categorical"]
                        if hasattr(cat_transformer, "get_feature_names_out"):
                            # Use stored feature names if available
                            stored_cat_features = preprocessing_info.get("feature_names_after", [])
                            if stored_cat_features:
                                # Filter for categorical features only
                                feature_names.extend(
                                    [
                                        f
                                        for f in stored_cat_features
                                        if any(f.startswith(c + "_") for c in categorical_cols)
                                    ],
                                )
                            else:
                                cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                                feature_names.extend(cat_features)
                        else:
                            # Fallback: reconstruct from categories
                            for i, col in enumerate(categorical_cols):
                                unique_vals = cat_transformer.categories_[i]
                                feature_names.extend([f"{col}_{val}" for val in unique_vals])

                    if numeric_cols:
                        feature_names.extend(numeric_cols)

                    # Validate feature count matches
                    if len(feature_names) != X_transformed.shape[1]:
                        logger.warning(
                            f"Feature count mismatch: expected {len(feature_names)} "
                            f"but got {X_transformed.shape[1]} from preprocessing. "
                            "Using generic feature names.",
                        )
                        # Fallback to generic names
                        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                    sanitized_feature_names = [
                        str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                        for name in feature_names
                    ]

                    X_processed = pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                    logger.info(f"Preprocessed {len(categorical_cols)} categorical columns with OneHotEncoder")
                    logger.info(f"Final feature count: {len(sanitized_feature_names)} (from {len(X.columns)} original)")
                    return X_processed

                except Exception as e:
                    logger.exception(f"Preprocessing failed: {e}")
                    raise ValueError(
                        f"Could not apply auto preprocessing: {e}. Ensure test data matches training data structure.",
                    )

            df = _apply_preprocessing_from_model_artifact(df, preprocessing_info)

        # Validate feature alignment if metadata available
        if expected_features is not None:
            available_features = list(df.columns)
            if set(expected_features) - set(available_features):
                missing = set(expected_features) - set(available_features)
                typer.secho(
                    f"Error: Model expects {len(expected_features)} features but data only has {len(available_features)} columns.\n"
                    f"Missing features: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}\n\n"
                    f"Model was trained on features: {expected_features[:5]}...\n"
                    f"Data has columns: {available_features[:5]}...\n\n"
                    f"Fix: Ensure data file has the same columns as the training data.\n"
                    f"• Check column names match exactly (case-sensitive)\n"
                    f"• Verify no columns were renamed or removed\n"
                    f"• Use the same data preprocessing pipeline",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(ExitCode.USER_ERROR)

            # Reorder columns to match training order
            df = df[expected_features]

        if instance < 0 or instance >= len(df):
            typer.secho(
                f"Error: Instance index {instance} is out of range in data file.\n"
                f"Data has {len(df)} rows (valid indices: 0-{len(df) - 1}).\n\n"
                f"Fix: Choose an instance index between 0 and {len(df) - 1}, or check that your data file contains the expected number of rows.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Get instance
        X_instance = df.iloc[[instance]].drop(columns=["target"], errors="ignore")
        feature_names = X_instance.columns.tolist()
        feature_values = X_instance.iloc[0]

        typer.echo(f"Generating SHAP explanations for instance {instance}...")

        # Auto-encode categorical columns for SHAP compatibility
        X_instance_encoded = X_instance.copy()
        categorical_cols = X_instance.select_dtypes(include=["object", "category", "string"]).columns

        if len(categorical_cols) > 0:
            typer.echo(f"Auto-encoding {len(categorical_cols)} categorical columns for SHAP compatibility...")
            from sklearn.preprocessing import LabelEncoder

            for col in categorical_cols:
                if X_instance_encoded[col].dtype in ["object", "category", "string"]:
                    # Convert to string first to handle any data types
                    X_instance_encoded[col] = X_instance_encoded[col].astype(str)
                    le = LabelEncoder()
                    X_instance_encoded[col] = le.fit_transform(X_instance_encoded[col])

        # Get prediction (with better error handling for feature mismatch)
        try:
            # Handle XGBoost Booster format specially
            if type(model_obj).__name__ == "Booster" or (
                hasattr(model_obj, "model") and type(model_obj.model).__name__ == "Booster"
            ):
                import xgboost as xgb

                booster = model_obj.model if hasattr(model_obj, "model") else model_obj
                X_dmatrix = xgb.DMatrix(X_instance_encoded)
                prediction = float(booster.predict(X_dmatrix)[0])
            else:
                # Convert DataFrame to numpy for compatibility
                X_numpy = X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded

                if hasattr(model_obj, "predict_proba"):
                    prediction = float(model_obj.predict_proba(X_numpy)[0, 1])
                else:
                    prediction = float(model_obj.predict(X_numpy)[0])
        except ValueError as e:
            error_msg = str(e)
            if "Too many missing features" in error_msg or (
                "features" in error_msg.lower() and "expecting" in error_msg.lower()
            ):
                # Extract feature counts from error message if available
                import re

                match = re.search(r"has (\d+) features.*expecting (\d+) features", error_msg)
                if match:
                    data_features, expected_count = match.groups()
                    feature_info = f"Model expects {expected_count} features but data has {data_features} columns"
                else:
                    feature_info = f"Feature count mismatch: {error_msg}"

                typer.secho(
                    f"❌ {feature_info}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\n📊 This usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data (e.g., one-hot encoded)")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("  • Model was saved without preprocessing metadata")
                typer.echo("\n✅ Solutions:")
                typer.echo("  1. Retrain and save model with preprocessing metadata:")
                typer.echo("     glassalpha audit --config your_config.yaml --save-model model.pkl")
                typer.echo("  2. Or use the exact preprocessed dataset that was used for training")
                typer.echo("  3. Or include preprocessing artifact path in model metadata")
                typer.echo("\n📚 For more help:")
                typer.echo("  • Preprocessing guide: https://glassalpha.com/guides/preprocessing/")
                typer.echo("  • Reason codes tutorial: https://glassalpha.com/guides/reason-codes/")
                raise typer.Exit(ExitCode.USER_ERROR) from None
            raise

        # Extract native model from wrapper if needed
        native_model = model_obj
        if hasattr(model_obj, "model"):
            # GlassAlpha wrapper - extract underlying model
            native_model = model_obj.model
            typer.echo(f"Extracted native model from wrapper: {type(native_model).__name__}")

        # Generate feature contributions (SHAP or coefficients fallback)
        shap_values = None
        try:
            # Try SHAP first (best for tree models and non-linear models)
            import shap

            try:
                typer.echo("  Computing TreeSHAP explanations...")
                typer.echo("    (This may take 10-30 seconds for tree models)")

                explainer = shap.TreeExplainer(native_model)

                # For XGBoost Booster, need to convert to DMatrix format
                if type(native_model).__name__ == "Booster":
                    import xgboost as xgb

                    X_shap = xgb.DMatrix(X_instance_encoded)
                else:
                    # Convert DataFrame to numpy for SHAP compatibility
                    X_shap = X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded

                typer.echo("    Computing SHAP values...", err=True)
                shap_values = explainer.shap_values(X_shap)

                # Handle multi-output case (binary classification)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class

                # Flatten to 1D
                if len(shap_values.shape) > 1:
                    shap_values = shap_values[0]

                typer.echo("    ✓ SHAP computation complete", err=True)
            except Exception as tree_error:
                # TreeSHAP failed, try KernelSHAP
                error_msg = str(tree_error).lower()
                if any(
                    phrase in error_msg
                    for phrase in [
                        "model type not yet supported",
                        "treeexplainer",
                        "does not support treeexplainer",
                        "linear model",
                        "logisticregression",
                    ]
                ):
                    typer.secho(
                        f"Note: TreeSHAP not compatible with {type(native_model).__name__}. Using KernelSHAP...",
                        fg=typer.colors.CYAN,
                    )
                    typer.echo("  Computing KernelSHAP explanations...")
                    typer.echo("    (This may take 30-60 seconds for linear models)")

                    # Load full dataset for background samples (KernelSHAP needs representative background)
                    # Use the full test data since we already loaded it
                    X_background = df.drop(columns=["target"], errors="ignore")

                    # Auto-encode categorical columns for background data
                    X_background_encoded = X_background.copy()
                    for col in categorical_cols:
                        if col in X_background_encoded.columns:
                            X_background_encoded[col] = X_background_encoded[col].astype(str)
                            from sklearn.preprocessing import LabelEncoder

                            le = LabelEncoder()
                            X_background_encoded[col] = le.fit_transform(X_background_encoded[col])

                    # Sample 100 background instances for KernelSHAP
                    background_sample = shap.sample(X_background_encoded, min(100, len(X_background_encoded)))

                    # Convert instance to numpy for compatibility
                    X_sample = (
                        X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded
                    )

                    typer.echo("    Initializing KernelExplainer...", err=True)
                    explainer = shap.KernelExplainer(model_obj.predict_proba, background_sample)

                    typer.echo("    Computing SHAP values...", err=True)
                    shap_values_raw = explainer.shap_values(X_sample)

                    # Handle different SHAP output formats for binary classification
                    # KernelExplainer returns [instances, features, classes] for predict_proba
                    # We want the positive class SHAP values for the single instance
                    if isinstance(shap_values_raw, list):
                        # Multi-output format: list of arrays, one per class
                        shap_values = shap_values_raw[1][0]  # Positive class, first instance
                    elif len(shap_values_raw.shape) == 3:
                        # 3D array format: [instances, features, classes]
                        shap_values = shap_values_raw[0, :, 1]  # First instance, positive class
                    elif len(shap_values_raw.shape) == 2:
                        # 2D array format: [instances, features] - already for single class
                        shap_values = shap_values_raw[0]  # First instance
                    else:
                        # 1D array: already the right format
                        shap_values = shap_values_raw

                    typer.echo("    ✓ KernelSHAP computation complete", err=True)
                else:
                    raise

        except ImportError:
            # SHAP not available - try coefficient-based fallback for linear models
            if hasattr(native_model, "coef_"):
                typer.secho(
                    "Note: SHAP not installed. Using coefficient-based explanations for linear model.",
                    fg=typer.colors.CYAN,
                )
                typer.echo("  For better explanations: pip install 'glassalpha[explain]'")
                typer.echo()

                # Use coefficients as feature importance
                coefficients = native_model.coef_[0] if len(native_model.coef_.shape) > 1 else native_model.coef_

                # Convert to contributions: coef * (feature_value - mean)
                # This approximates local explanation for linear models
                X_numpy = X_instance_encoded.values[0] if hasattr(X_instance_encoded, "values") else X_instance_encoded
                shap_values = coefficients * X_numpy
            else:
                typer.secho(
                    "Error: SHAP not installed and model doesn't support coefficient extraction.",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo()
                typer.echo("Solutions:")
                typer.echo("  1. Install SHAP: pip install 'glassalpha[explain]'")
                typer.echo("  2. Use a model with coefficients (LogisticRegression)")
                raise typer.Exit(ExitCode.USER_ERROR)
        except Exception as e:
            if "Too many missing features" in str(e):
                typer.secho(
                    f"Error: Model expects {len(expected_features) if expected_features else 'unknown'} features but data has {len(X_instance_encoded.columns)} columns",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nThis usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("\nSolutions:")
                typer.echo("  1. Use the same dataset that was used for training")
                typer.echo("  2. Apply the same preprocessing (encoding) that was used during training")
                typer.echo("  3. Retrain the model on raw data if preprocessing isn't available")
                typer.echo("\nFor help with preprocessing, see: https://glassalpha.com/guides/preprocessing/")
            else:
                typer.secho(
                    f"Error generating explanations: {e}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nTip: Ensure model is compatible (XGBoost, LightGBM, RandomForest, LogisticRegression)")
                typer.echo("For linear models without SHAP: pip install 'glassalpha[explain]'")
            raise typer.Exit(ExitCode.USER_ERROR) from None

        # Extract reason codes
        from glassalpha.explain.reason_codes import extract_reason_codes, format_adverse_action_notice

        typer.echo("Extracting top-N negative contributions...")
        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=instance,
            prediction=prediction,
            threshold=threshold,
            top_n=top_n,
            protected_attributes=protected_attributes,
            seed=seed,
        )

        # Format output
        if format == "json":
            output_dict = {
                "instance_id": result.instance_id,
                "prediction": result.prediction,
                "decision": result.decision,
                "reason_codes": [
                    {
                        "rank": code.rank,
                        "feature": code.feature,
                        "contribution": code.contribution,
                        "feature_value": code.feature_value,
                    }
                    for code in result.reason_codes
                ],
                "excluded_features": result.excluded_features,
                "timestamp": result.timestamp,
                "model_hash": result.model_hash,
                "seed": result.seed,
            }
            output_text = json.dumps(output_dict, indent=2, sort_keys=True)
        else:
            # Text format (ECOA notice)
            output_text = format_adverse_action_notice(
                result=result,
                organization=organization,
                contact_info=contact_info,
            )

        # Write or print output
        if output:
            output.write_text(output_text)
            typer.secho(
                "\n✅ Reason codes generated successfully!",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"Output: {output}")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(output_text)
            typer.echo("=" * 60)

        # Show summary
        typer.echo(f"\nInstance: {result.instance_id}")
        typer.echo(f"Prediction: {result.prediction:.1%}")
        typer.echo(f"Decision: {result.decision.upper()}")
        typer.echo(f"Reason codes: {len(result.reason_codes)}")
        if result.excluded_features:
            typer.echo(f"Protected attributes excluded: {len(result.excluded_features)}")

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except KeyError as e:
        # Feature name mismatch - likely using raw data instead of preprocessed
        typer.secho(f"\n❌ Error: Column {e} not found in data", fg=typer.colors.RED, err=True)
        typer.echo()
        typer.echo("💡 Common issue: You may be using raw data instead of preprocessed test data.")
        typer.echo()
        typer.echo("Reason codes require the PREPROCESSED data saved during audit generation:")
        typer.echo("  ✅ Correct: --data models/test_data.csv")
        typer.echo("  ❌ Wrong:   --data original_data.csv")
        typer.echo()
        typer.echo("📋 Solutions:")
        typer.echo()
        typer.echo("  1. Use the test_data.csv generated by audit:")
        typer.echo(f"     glassalpha reasons --model {model} --data models/test_data.csv --instance {instance}")
        typer.echo()
        typer.echo("  2. Run audit to generate test_data.csv:")
        typer.echo("     glassalpha audit --config config.yaml")
        typer.echo()
        typer.echo("📖 For more help:")
        typer.echo("   https://glassalpha.com/guides/reason-codes/#data-format-requirements")
        typer.echo()
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        # Note: Can't reliably check sys.argv here due to Typer's exception handling
        # Log exception regardless - users can check logs if needed
        logger.exception("Reason code generation failed")
        typer.secho(f"Failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None


def recourse(  # pragma: no cover
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl",
        exists=True,
        file_okay=True,
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to test data file (CSV). Auto-generated if missing using built-in dataset.",
        file_okay=True,
    ),
    instance: int = typer.Option(
        ...,
        "--instance",
        "-i",
        help="Row index of instance to explain (0-based)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to recourse configuration YAML",
        file_okay=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for output recommendations file (JSON, defaults to stdout)",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Decision threshold for approved/denied",
    ),
    top_n: int = typer.Option(
        5,
        "--top-n",
        "-n",
        help="Number of counterfactual recommendations to generate",
    ),
    force_recourse: bool = typer.Option(
        False,
        "--force-recourse",
        help="Generate recourse recommendations even for approved instances (for testing)",
    ),
):
    """Generate ECOA-compliant counterfactual recourse recommendations.

    This command generates feasible counterfactual recommendations with policy constraints
    for individuals receiving adverse decisions. Supports immutable features, monotonic
    constraints, and cost-weighted optimization.

    Requirements:
        - Trained model with SHAP-compatible architecture
        - Test dataset with same features as training
        - Instance index to explain (must be denied: prediction < threshold)
        - Configuration file with policy constraints (recommended)

    Examples:
        # Generate recourse for denied instance
        glassalpha recourse \\
            --model models/german_credit.pkl \\
            --data data/test.csv \\
            --instance 42 \\
            --config configs/recourse_german_credit.yaml \\
            --output recourse/instance_42.json

        # With custom threshold and top-N
        glassalpha recourse -m model.pkl -d test.csv -i 10 -c config.yaml --top-n 3

        # Output to stdout
        glassalpha recourse -m model.pkl -d test.csv -i 5 -c config.yaml

    Configuration File:
        The config file should include:
        - recourse.immutable_features: list of features that cannot be changed
        - recourse.monotonic_constraints: directional constraints (increase_only, decrease_only)
        - recourse.cost_function: cost function for optimization (weighted_l1)
        - data.protected_attributes: list of protected attributes to exclude
        - reproducibility.random_seed: seed for deterministic results

    Model Compatibility:
        Recourse works best with sklearn-compatible models:
        ✅ logistic_regression, linear_regression, random_forest (sklearn)
        [WARN] xgboost, lightgbm (limited support - known issues with feature modification)

        For XGBoost models, consider using 'glassalpha reasons' instead for ECOA-compliant
        adverse action notices. See: https://glassalpha.com/guides/recourse/#known-limitations

    """
    # Lazy imports - only when function is actually called
    import json

    import joblib

    try:
        import pandas as pd  # Lazy import for performance

        # Load configuration
        immutable_features: list[str] = []
        monotonic_constraints: dict[str, str] = {}
        feature_costs: dict[str, float] = {}
        feature_bounds: dict[str, tuple[float, float]] = {}
        seed = 42

        if config and config.exists():
            from glassalpha.config import load_config

            cfg = load_config(config)

            # Load recourse config
            if hasattr(cfg, "recourse"):
                immutable_features = list(getattr(cfg.recourse, "immutable_features", []))
                raw_constraints = getattr(cfg.recourse, "monotonic_constraints", {})
                # Convert monotonic constraints to dict[str, str] for API
                monotonic_constraints = {str(k): str(v) for k, v in raw_constraints.items()}

            # Load seed
            seed = getattr(cfg.reproducibility, "random_seed", 42) if hasattr(cfg, "reproducibility") else 42
        else:
            typer.secho(
                "[WARN] Warning: No config provided. Using default policy with common protected attributes as immutable.",
                fg=typer.colors.YELLOW,
            )
            typer.echo()
            typer.echo("  Default immutable features: gender, race, age, sex, ethnicity (and one-hot variants)")
            typer.echo("  To customize, create a config file with recourse.immutable_features")
            typer.echo()
            # Use common protected attributes as default immutables
            # This includes the base names and common one-hot patterns
            default_immutable_patterns = [
                "gender",
                "sex",
                "race",
                "ethnicity",
                "age",
                "age_years",
                "age_group",
                "foreign_worker",
                "nationality",
                "national_origin",
            ]
            # We'll filter these by actual column names later
            immutable_features = default_immutable_patterns

        # Check if data file exists, look for model-compatible test data first
        if not data.exists():
            # Smart path detection: Check if model directory has test_data.csv
            model_dir = model.parent
            model_test_data = model_dir / "test_data.csv"

            if model_test_data.exists():
                typer.echo(f"💡 Using test data from model directory: {model_test_data}")
                typer.echo("   (This data matches the model's expected feature structure)")
                typer.echo()
                data = model_test_data
            else:
                # Data file not found - provide clear error with solutions
                typer.secho(f"❌ Error: Test data file not found: {data}", fg=typer.colors.RED, err=True)
                typer.echo()
                typer.echo("The reason codes/recourse commands require test data that matches")
                typer.echo("the model's expected feature structure (including preprocessing).")
                typer.echo()
                typer.echo("📋 Solutions:")
                typer.echo()
                typer.echo("  1. Use test data saved during audit generation:")
                typer.echo(f"     --data {model_test_data}")
                typer.echo()
                typer.echo("  2. Run audit to generate and save test data:")
                typer.echo("     glassalpha audit --config config.yaml")
                typer.echo("     (Automatically saves to models/test_data.csv)")
                typer.echo()
                typer.echo("  3. Provide your own test CSV with same features as training")
                typer.echo("     (after preprocessing, including one-hot encoding)")
                typer.echo()
                typer.echo("📖 For more help:")
                typer.echo("   https://glassalpha.com/guides/reason-codes/")
                typer.echo()
                raise typer.Exit(ExitCode.USER_ERROR)

        typer.echo(f"Loading model from: {model}")
        # Use joblib for loading (matches saving with joblib.dump in audit command)
        import joblib

        loaded = joblib.load(model)

        # Check for companion .meta.json file
        meta_path = Path(model).with_suffix(".meta.json")
        model_metadata = None
        if meta_path.exists():
            try:
                model_metadata = json.loads(meta_path.read_text())
                typer.echo(f"Found model metadata: {meta_path}")
            except Exception as e:
                logger.warning(f"Failed to load model metadata from {meta_path}: {e}")

        # Handle both old format (model only) and new format (dict with metadata)
        if isinstance(loaded, dict) and "model" in loaded:
            model_obj = loaded["model"]
            expected_features = loaded.get("feature_names")
            preprocessing_info = loaded.get("preprocessing")
        else:
            model_obj = loaded
            expected_features = model_metadata.get("feature_names") if model_metadata else None
            preprocessing_info = None
            if model_metadata and model_metadata.get("preprocessing"):
                preprocessing_dict = model_metadata.get("preprocessing")
                if preprocessing_dict.get("artifact_path"):
                    preprocessing_info = {
                        "mode": "artifact",
                        "artifact_path": preprocessing_dict["artifact_path"],
                        "feature_names_after": expected_features,
                    }

        typer.echo(f"Loading data from: {data}")
        df = pd.read_csv(data)

        # Check if data is already preprocessed by comparing with expected features
        data_is_preprocessed = False
        if preprocessing_info:
            expected_features_list = preprocessing_info.get("feature_names_after", [])
            if expected_features_list:
                # Check column overlap - if >90% match, assume already preprocessed
                matching_cols = len(set(df.columns).intersection(set(expected_features_list)))
                if matching_cols > len(expected_features_list) * 0.9:
                    data_is_preprocessed = True
                    typer.echo("✓ Data is already preprocessed (matches model's expected features)")

        # Apply preprocessing if available and data is NOT already preprocessed
        if preprocessing_info and not data_is_preprocessed:
            typer.echo("Applying preprocessing to match model training...")

            def _apply_preprocessing_from_model_artifact(
                X: "pd.DataFrame",
                preprocessing_info: dict | None,
            ) -> "pd.DataFrame":
                """Apply the same preprocessing that was used during model training."""
                if preprocessing_info is None:
                    return X
                mode = preprocessing_info.get("mode", "auto")

                if mode == "artifact":
                    # Load preprocessing artifact and apply it
                    artifact_path = preprocessing_info.get("artifact_path")
                    if artifact_path:
                        try:
                            import joblib

                            preprocessor = joblib.load(artifact_path)
                            logger.info(f"Applying preprocessing artifact from: {artifact_path}")

                            # Apply preprocessing (assumes target column is not in X)
                            # Use transform, not fit_transform to match training exactly
                            X_transformed = preprocessor.transform(X)

                            # Get expected feature names from preprocessing info or artifact
                            expected_features = preprocessing_info.get("feature_names_after")
                            if expected_features:
                                # Use stored feature names for consistency
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in expected_features
                                ]
                            else:
                                # Fallback: get from artifact
                                feature_names = preprocessor.get_feature_names_out()
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in feature_names
                                ]

                            # Validate feature count matches expectations
                            if len(sanitized_feature_names) != X_transformed.shape[1]:
                                raise ValueError(
                                    f"Feature count mismatch: expected {len(sanitized_feature_names)} "
                                    f"features but got {X_transformed.shape[1]} from preprocessing",
                                )

                            # Return as DataFrame with proper column names
                            return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                        except Exception as e:
                            logger.exception(f"Failed to apply preprocessing artifact: {e}")
                            raise ValueError(
                                f"Could not apply preprocessing artifact: {e}. "
                                "Ensure the preprocessing artifact is accessible and matches the training data.",
                            )
                    else:
                        raise ValueError("Artifact mode specified but no artifact_path provided in model metadata")
                elif mode == "auto":
                    return _apply_auto_preprocessing_from_metadata(X, preprocessing_info)
                else:
                    raise ValueError(f"Unknown preprocessing mode: {mode}")

            def _apply_auto_preprocessing_from_metadata(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                # Check if data is already preprocessed (e.g., loaded from test_data.csv)
                # If X has more columns than expected raw features, it's likely already one-hot encoded
                expected_raw_cols = len(categorical_cols) + len(numeric_cols)
                if len(X.columns) > expected_raw_cols:
                    logger.debug(
                        f"Data appears already preprocessed ({len(X.columns)} cols vs {expected_raw_cols} expected raw). Skipping preprocessing."
                    )
                    return X

                logger.debug(
                    f"Applying auto preprocessing from metadata: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    expected_cols = sorted(categorical_cols + numeric_cols)
                    actual_cols = sorted(X.columns)
                    missing_list = sorted(missing_cols)

                    # Show truncated lists for readability
                    missing_display = ", ".join(missing_list[:10])
                    if len(missing_list) > 10:
                        missing_display += f"... (+{len(missing_list) - 10} more)"

                    expected_display = ", ".join(expected_cols[:10])
                    if len(expected_cols) > 10:
                        expected_display += f"... (+{len(expected_cols) - 10} more)"

                    raise ValueError(
                        f"Test data missing {len(missing_cols)} columns.\n\n"
                        f"Missing columns:\n  {missing_display}\n\n"
                        f"Expected {len(expected_cols)} columns:\n  {expected_display}\n\n"
                        f"Actual {len(actual_cols)} columns:\n  {', '.join(actual_cols)}\n\n"
                        f"Fix options:\n"
                        f"  1. Use original training data (with all features)\n"
                        f"  2. Re-run audit and save test data: glassalpha audit --save-test-data\n"
                        f"  3. Provide config for auto-loading: --config audit_config.yaml"
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                # Use training-time preprocessing info to recreate the same transformation
                # We need to fit on training-like data or use stored parameters
                # For now, fit on the test data but this may not be identical to training
                # TODO: Store and use training preprocessor parameters for perfect reproducibility

                X_transformed = preprocessor.fit_transform(X)
                feature_names = []

                # Reconstruct feature names based on training metadata
                if categorical_cols:
                    cat_transformer = preprocessor.named_transformers_["categorical"]
                    if hasattr(cat_transformer, "get_feature_names_out"):
                        # Use stored feature names if available
                        stored_cat_features = preprocessing_info.get("feature_names_after", [])
                        if stored_cat_features:
                            # Filter for categorical features only
                            feature_names.extend(
                                [
                                    f
                                    for f in stored_cat_features
                                    if any(f.startswith(c + "_") for c in categorical_cols)
                                ],
                            )
                        else:
                            cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                            feature_names.extend(cat_features)
                    else:
                        # Fallback: reconstruct from categories
                        for i, col in enumerate(categorical_cols):
                            unique_vals = cat_transformer.categories_[i]
                            feature_names.extend([f"{col}_{val}" for val in unique_vals])

                if numeric_cols:
                    feature_names.extend(numeric_cols)

                # Validate feature count matches
                if len(feature_names) != X_transformed.shape[1]:
                    logger.warning(
                        f"Feature count mismatch: expected {len(feature_names)} "
                        f"but got {X_transformed.shape[1]} from preprocessing. "
                        "Using generic feature names.",
                    )
                    # Fallback to generic names
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                sanitized_feature_names = [
                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                    for name in feature_names
                ]

                return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)

            def _apply_auto_preprocessing(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                logger.debug(
                    f"Applying auto preprocessing: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    expected_cols = sorted(categorical_cols + numeric_cols)
                    actual_cols = sorted(X.columns)
                    missing_list = sorted(missing_cols)

                    # Show truncated lists for readability
                    missing_display = ", ".join(missing_list[:10])
                    if len(missing_list) > 10:
                        missing_display += f"... (+{len(missing_list) - 10} more)"

                    expected_display = ", ".join(expected_cols[:10])
                    if len(expected_cols) > 10:
                        expected_display += f"... (+{len(expected_cols) - 10} more)"

                    raise ValueError(
                        f"Test data missing {len(missing_cols)} columns.\n\n"
                        f"Missing columns:\n  {missing_display}\n\n"
                        f"Expected {len(expected_cols)} columns:\n  {expected_display}\n\n"
                        f"Actual {len(actual_cols)} columns:\n  {', '.join(actual_cols)}\n\n"
                        f"Fix options:\n"
                        f"  1. Use original training data (with all features)\n"
                        f"  2. Re-run audit and save test data: glassalpha audit --save-test-data\n"
                        f"  3. Provide config for auto-loading: --config audit_config.yaml"
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                try:
                    X_transformed = preprocessor.fit_transform(X)
                    feature_names = []

                    # Reconstruct feature names based on training metadata
                    if categorical_cols:
                        cat_transformer = preprocessor.named_transformers_["categorical"]
                        if hasattr(cat_transformer, "get_feature_names_out"):
                            # Use stored feature names if available
                            stored_cat_features = preprocessing_info.get("feature_names_after", [])
                            if stored_cat_features:
                                # Filter for categorical features only
                                feature_names.extend(
                                    [
                                        f
                                        for f in stored_cat_features
                                        if any(f.startswith(c + "_") for c in categorical_cols)
                                    ],
                                )
                            else:
                                cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                                feature_names.extend(cat_features)
                        else:
                            # Fallback: reconstruct from categories
                            for i, col in enumerate(categorical_cols):
                                unique_vals = cat_transformer.categories_[i]
                                feature_names.extend([f"{col}_{val}" for val in unique_vals])

                    if numeric_cols:
                        feature_names.extend(numeric_cols)

                    # Validate feature count matches
                    if len(feature_names) != X_transformed.shape[1]:
                        logger.warning(
                            f"Feature count mismatch: expected {len(feature_names)} "
                            f"but got {X_transformed.shape[1]} from preprocessing. "
                            "Using generic feature names.",
                        )
                        # Fallback to generic names
                        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                    sanitized_feature_names = [
                        str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                        for name in feature_names
                    ]

                    X_processed = pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                    logger.info(f"Preprocessed {len(categorical_cols)} categorical columns with OneHotEncoder")
                    logger.info(f"Final feature count: {len(sanitized_feature_names)} (from {len(X.columns)} original)")
                    return X_processed

                except Exception as e:
                    logger.exception(f"Preprocessing failed: {e}")
                    raise ValueError(
                        f"Could not apply auto preprocessing: {e}. Ensure test data matches training data structure.",
                    )

            df = _apply_preprocessing_from_model_artifact(df, preprocessing_info)

        # Validate feature alignment if metadata available
        if expected_features is not None:
            available_features = list(df.columns)
            if set(expected_features) - set(available_features):
                missing = set(expected_features) - set(available_features)
                typer.secho(
                    f"Error: Model expects {len(expected_features)} features but data only has {len(available_features)} columns.\n"
                    f"Missing features: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}\n\n"
                    f"Model was trained on features: {expected_features[:5]}...\n"
                    f"Data has columns: {available_features[:5]}...\n\n"
                    f"Fix: Ensure data file has the same columns as the training data.\n"
                    f"• Check column names match exactly (case-sensitive)\n"
                    f"• Verify no columns were renamed or removed\n"
                    f"• Use the same data preprocessing pipeline",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(ExitCode.USER_ERROR)

            # Reorder columns to match training order
            df = df[expected_features]

        if instance < 0 or instance >= len(df):
            typer.secho(
                f"Error: Instance index {instance} is out of range in data file.\n"
                f"Data has {len(df)} rows (valid indices: 0-{len(df) - 1}).\n\n"
                f"Fix: Choose an instance index between 0 and {len(df) - 1}, or check that your data file contains the expected number of rows.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Get instance
        X_instance = df.iloc[[instance]].drop(columns=["target"], errors="ignore")
        feature_names = X_instance.columns.tolist()
        feature_values_series = X_instance.iloc[0]

        typer.echo(f"Generating SHAP explanations for instance {instance}...")

        # Auto-encode categorical columns for SHAP compatibility
        X_instance_encoded = X_instance.copy()
        categorical_cols = X_instance.select_dtypes(include=["object", "category", "string"]).columns

        if len(categorical_cols) > 0:
            typer.echo(f"Auto-encoding {len(categorical_cols)} categorical columns for SHAP compatibility...")
            from sklearn.preprocessing import LabelEncoder

            for col in categorical_cols:
                if X_instance_encoded[col].dtype in ["object", "category", "string"]:
                    # Convert to string first to handle any data types
                    X_instance_encoded[col] = X_instance_encoded[col].astype(str)
                    le = LabelEncoder()
                    X_instance_encoded[col] = le.fit_transform(X_instance_encoded[col])

        # Get prediction (with better error handling for feature mismatch)
        try:
            # Handle XGBoost Booster format specially
            if type(model_obj).__name__ == "Booster" or (
                hasattr(model_obj, "model") and type(model_obj.model).__name__ == "Booster"
            ):
                import xgboost as xgb

                booster = model_obj.model if hasattr(model_obj, "model") else model_obj
                X_dmatrix = xgb.DMatrix(X_instance_encoded)
                prediction = float(booster.predict(X_dmatrix)[0])
            elif hasattr(model_obj, "predict_proba"):
                prediction = float(model_obj.predict_proba(X_instance_encoded)[0, 1])
            else:
                prediction = float(model_obj.predict(X_instance_encoded)[0])
        except ValueError as e:
            error_msg = str(e)
            if "Too many missing features" in error_msg or (
                "features" in error_msg.lower() and "expecting" in error_msg.lower()
            ):
                # Extract feature counts from error message if available
                import re

                match = re.search(r"has (\d+) features.*expecting (\d+) features", error_msg)
                if match:
                    data_features, expected_count = match.groups()
                    feature_info = f"Model expects {expected_count} features but data has {data_features} columns"
                else:
                    feature_info = f"Feature count mismatch: {error_msg}"

                typer.secho(
                    f"❌ {feature_info}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\n📊 This usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data (e.g., one-hot encoded)")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("  • Model was saved without preprocessing metadata")
                typer.echo("\n✅ Solutions:")
                typer.echo("  1. Retrain and save model with preprocessing metadata:")
                typer.echo("     glassalpha audit --config your_config.yaml --save-model model.pkl")
                typer.echo("  2. Or use the exact preprocessed dataset that was used for training")
                typer.echo("  3. Or include preprocessing artifact path in model metadata")
                typer.echo("\n📚 For more help:")
                typer.echo("  • Preprocessing guide: https://glassalpha.com/guides/preprocessing/")
                typer.echo("  • Recourse tutorial: https://glassalpha.com/guides/recourse/")
                raise typer.Exit(ExitCode.USER_ERROR) from None
            raise

        # Check if instance is already approved
        if prediction >= threshold and not force_recourse:
            typer.secho(
                f"\nInstance {instance} is already approved (prediction={prediction:.1%} >= threshold={threshold:.1%})",
                fg=typer.colors.GREEN,
            )
            typer.echo("No recourse needed.")
            typer.echo("\nTo generate recourse anyway (for testing), use --force-recourse flag.")
            raise typer.Exit(ExitCode.SUCCESS)

        # Extract native model from wrapper if needed
        native_model = model_obj
        if hasattr(model_obj, "model"):
            # GlassAlpha wrapper - extract underlying model
            native_model = model_obj.model
            typer.echo(f"Extracted native model from wrapper: {type(native_model).__name__}")

        # Generate SHAP values (use TreeSHAP for tree models)
        try:
            import shap

            typer.echo("  Computing TreeSHAP explanations...")
            typer.echo("    (This may take 10-30 seconds for tree models)")

            explainer = shap.TreeExplainer(native_model)

            # For XGBoost Booster, need to convert to DMatrix format
            if type(native_model).__name__ == "Booster":
                import xgboost as xgb

                X_shap = xgb.DMatrix(X_instance_encoded)
            else:
                # Convert DataFrame to numpy for SHAP compatibility
                X_shap = X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded

            typer.echo("    Computing SHAP values...")
            shap_values = explainer.shap_values(X_shap)
            typer.echo("    ✓ SHAP computation complete")

            # Handle multi-output case (binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Flatten to 1D
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

        except Exception as e:
            if "Too many missing features" in str(e):
                typer.secho(
                    f"Error: Model expects {len(expected_features) if expected_features else 'unknown'} features but data has {len(X_instance_encoded.columns)} columns",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nThis usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("\nSolutions:")
                typer.echo("  1. Use the same dataset that was used for training")
                typer.echo("  2. Apply the same preprocessing (encoding) that was used during training")
                typer.echo("  3. Retrain the model on raw data if preprocessing isn't available")
                typer.echo("\nFor help with preprocessing, see: https://glassalpha.com/guides/preprocessing/")
                raise typer.Exit(ExitCode.USER_ERROR) from None
            # Check if this is a TreeSHAP compatibility error for non-tree models
            error_msg = str(e).lower()
            if any(
                phrase in error_msg
                for phrase in [
                    "model type not yet supported",
                    "treeexplainer",
                    "does not support treeexplainer",
                    "linear model",
                    "logisticregression",
                ]
            ):
                typer.secho(
                    f"Warning: TreeSHAP not compatible with {type(native_model).__name__}. Using KernelSHAP instead...",
                    fg=typer.colors.YELLOW,
                )
                # Fallback to KernelSHAP (use original model_obj for predict interface)
                import shap

                # Load full dataset for background samples
                X_background = df.drop(columns=["target"], errors="ignore")

                # Auto-encode categorical columns for background data
                X_background_encoded = X_background.copy()
                for col in categorical_cols:
                    if col in X_background_encoded.columns:
                        X_background_encoded[col] = X_background_encoded[col].astype(str)
                        from sklearn.preprocessing import LabelEncoder

                        le = LabelEncoder()
                        X_background_encoded[col] = le.fit_transform(X_background_encoded[col])

                # Sample 100 background instances
                background_sample = shap.sample(X_background_encoded, min(100, len(X_background_encoded)))

                explainer = shap.KernelExplainer(model_obj.predict_proba, background_sample)
                shap_values_raw = explainer.shap_values(X_instance_encoded)

                # Handle different SHAP output formats
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[1][0]  # Positive class, first instance
                elif len(shap_values_raw.shape) == 3:
                    shap_values = shap_values_raw[0, :, 1]  # First instance, positive class
                else:
                    shap_values = shap_values_raw[0]  # First instance
            else:
                typer.secho(
                    f"Error generating SHAP values: {e}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nTip: Ensure model is TreeSHAP-compatible (XGBoost, LightGBM, RandomForest)")
                typer.echo("If using a wrapped model, save the native model directly instead.")
                raise typer.Exit(ExitCode.USER_ERROR) from None

        typer.echo("Generating counterfactual recommendations...")

        # Build policy constraints
        from glassalpha.explain.policy import PolicyConstraints

        # Filter immutable features to only include actual column names
        # Also include one-hot encoded variants (e.g., gender_male, gender_female)
        actual_immutable_features = []
        for pattern in immutable_features:
            # Check for exact match
            if pattern in feature_names:
                actual_immutable_features.append(pattern)
            # Check for one-hot encoded variants (pattern_*)
            else:
                matching_cols = [name for name in feature_names if name.startswith(f"{pattern}_")]
                actual_immutable_features.extend(matching_cols)

        # Show what's being used
        if actual_immutable_features:
            typer.echo(
                f"  Immutable features ({len(actual_immutable_features)}): {', '.join(actual_immutable_features[:5])}{'...' if len(actual_immutable_features) > 5 else ''}"
            )
        else:
            typer.secho(
                "[WARN] Warning: No immutable features found. All features are mutable.",
                fg=typer.colors.YELLOW,
            )
            typer.echo("  This may generate unrealistic recommendations.")

        policy = PolicyConstraints(
            immutable_features=actual_immutable_features,
            monotonic_constraints=monotonic_constraints,
            feature_costs=feature_costs if feature_costs else dict.fromkeys(feature_names, 1.0),
            feature_bounds=feature_bounds,
        )

        # Wrap XGBoost Booster for predict_proba compatibility
        if type(model_obj).__name__ == "Booster" or (
            hasattr(model_obj, "model") and type(model_obj.model).__name__ == "Booster"
        ):
            import numpy as np
            import xgboost as xgb

            class BoosterWrapper:
                """Wrapper to make XGBoost Booster compatible with predict_proba interface."""

                def __init__(self, booster):
                    self.booster = booster

                def predict_proba(self, X):
                    """Predict probabilities using DMatrix."""
                    dmatrix = xgb.DMatrix(X)
                    pred = self.booster.predict(dmatrix)
                    # Return in (n_samples, n_classes) format
                    return np.column_stack([1 - pred, pred])

            booster = model_obj.model if hasattr(model_obj, "model") else model_obj
            wrapped_model = BoosterWrapper(booster)
        else:
            wrapped_model = model_obj

        # Generate recourse
        from glassalpha.explain.recourse import generate_recourse

        typer.echo()
        typer.echo("  Generating counterfactual recommendations...")
        typer.echo(f"    (Finding top {top_n} actionable changes)")

        result = generate_recourse(
            model=wrapped_model,
            feature_values=feature_values_series,
            shap_values=shap_values,
            feature_names=feature_names,
            instance_id=instance,
            original_prediction=prediction,
            threshold=threshold,
            policy_constraints=policy,
            top_n=top_n,
            seed=seed,
        )

        typer.echo("    ✓ Recourse generation complete")
        typer.echo()

        # Format output as JSON
        output_dict = {
            "instance_id": result.instance_id,
            "original_prediction": result.original_prediction,
            "threshold": result.threshold,
            "recommendations": [
                {
                    "rank": rec.rank,
                    "feature_changes": {
                        feature: {"old": old_val, "new": new_val}
                        for feature, (old_val, new_val) in rec.feature_changes.items()
                    },
                    "total_cost": rec.total_cost,
                    "predicted_probability": rec.predicted_probability,
                    "feasible": rec.feasible,
                }
                for rec in result.recommendations
            ],
            "policy_constraints": {
                "immutable_features": result.policy_constraints.immutable_features,
                "monotonic_constraints": result.policy_constraints.monotonic_constraints,
            },
            "seed": result.seed,
            "total_candidates": result.total_candidates,
            "feasible_candidates": result.feasible_candidates,
        }
        output_text = json.dumps(output_dict, indent=2, sort_keys=True)

        # Write or print output
        if output:
            output.write_text(output_text)
            typer.secho(
                "\n✅ Recourse recommendations generated successfully!",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"Output: {output}")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(output_text)
            typer.echo("=" * 60)

        # Show summary
        typer.echo(f"\nInstance: {result.instance_id}")
        typer.echo(f"Original prediction: {result.original_prediction:.1%}")
        typer.echo(f"Threshold: {result.threshold:.1%}")
        typer.echo(f"Recommendations: {len(result.recommendations)}")
        typer.echo(f"Total candidates evaluated: {result.total_candidates}")
        typer.echo(f"Feasible candidates: {result.feasible_candidates}")

        if len(result.recommendations) == 0:
            # Provide specific guidance based on the situation
            if result.original_prediction >= result.threshold:
                typer.secho(
                    "\n✅ Instance already approved - no changes needed",
                    fg=typer.colors.GREEN,
                )
                typer.echo(f"   Prediction: {result.original_prediction:.1%} >= threshold {result.threshold:.1%}")
            elif result.total_candidates == 0:
                typer.secho(
                    "\n[WARN] Could not generate recourse candidates",
                    fg=typer.colors.YELLOW,
                )
                typer.echo("\nPossible reasons:")
                typer.echo("  • All features marked as immutable (nothing can be changed)")
                typer.echo("  • Feature bounds too restrictive")
                typer.echo("  • Model does not support feature modification")
                typer.echo("\n[TIP] Try:")
                typer.echo("  • Review immutable_features in config")
                typer.echo("  • Check feature bounds and constraints")
                typer.echo("  • Use 'glassalpha reasons' for simpler explanations")
            else:
                typer.secho(
                    f"\n[WARN] No feasible recourse found (evaluated {result.total_candidates} candidates)",
                    fg=typer.colors.YELLOW,
                )
                typer.echo("\n[TIP] Try:")
                typer.echo("  • Relax monotonic constraints (allow more feature directions)")
                typer.echo("  • Reduce immutable features (allow more features to change)")
                typer.echo("  • Increase feature change bounds (allow larger changes)")
                typer.echo("  • Simplify cost function")
                typer.echo("\n📖 See: https://glassalpha.com/guides/recourse/#known-limitations")

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        # Log exception for debugging (but not as ERROR if it's just a clean exit)
        if not isinstance(e, (typer.Exit, SystemExit)):
            logger.exception("Recourse generation failed")
        typer.secho(f"Failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
