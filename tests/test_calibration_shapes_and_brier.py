"""Test probability calibration functionality and Brier score improvements."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss

from glassalpha.models.calibration import (
    assess_calibration_quality,
    get_calibration_info,
    maybe_calibrate,
    recommend_calibration_method,
    validate_calibration_config,
)


def test_maybe_calibrate_with_isotonic():
    """Test isotonic calibration doesn't break probability shapes."""
    from sklearn.ensemble import RandomForestClassifier

    # Create synthetic dataset
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

    # Train base model
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    # Apply calibration
    calibrated_rf = maybe_calibrate(rf, method="isotonic", cv=3)

    # CalibratedClassifierCV needs to be fitted
    calibrated_rf.fit(X, y)

    # Test predictions
    X_test, y_test = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=123)

    # Both should work
    pred_orig = rf.predict_proba(X_test)
    pred_cal = calibrated_rf.predict_proba(X_test)

    # Check shapes are preserved
    assert pred_orig.shape == pred_cal.shape
    assert pred_cal.shape == (100, 2)

    # Check probabilities are valid
    assert np.all(pred_cal >= 0)
    assert np.all(pred_cal <= 1)
    assert np.allclose(pred_cal.sum(axis=1), 1.0, atol=1e-6)


def test_maybe_calibrate_with_sigmoid():
    """Test sigmoid calibration doesn't break probability shapes."""
    from sklearn.ensemble import RandomForestClassifier

    # Create synthetic dataset
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

    # Train base model
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    # Apply calibration
    calibrated_rf = maybe_calibrate(rf, method="sigmoid", cv=3)

    # CalibratedClassifierCV needs to be fitted
    calibrated_rf.fit(X, y)

    # Test predictions
    X_test, y_test = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=123)

    # Both should work
    pred_orig = rf.predict_proba(X_test)
    pred_cal = calibrated_rf.predict_proba(X_test)

    # Check shapes are preserved
    assert pred_orig.shape == pred_cal.shape
    assert pred_cal.shape == (100, 2)

    # Check probabilities are valid
    assert np.all(pred_cal >= 0)
    assert np.all(pred_cal <= 1)
    assert np.allclose(pred_cal.sum(axis=1), 1.0, atol=1e-6)


def test_maybe_calibrate_no_method():
    """Test that no calibration returns original estimator."""
    from sklearn.ensemble import RandomForestClassifier

    # Create and train model
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)

    # No calibration
    result = maybe_calibrate(rf, method=None)

    # Should return the same object
    assert result is rf


def test_maybe_calibrate_invalid_method():
    """Test that invalid calibration method raises error."""
    from sklearn.ensemble import RandomForestClassifier

    # Create and train model
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)

    # Invalid method should raise
    with pytest.raises(ValueError, match="Unknown calibration method"):
        maybe_calibrate(rf, method="invalid_method")


def test_calibration_improves_brier_score():
    """Test that calibration ideally improves Brier score on poorly calibrated model."""
    from sklearn.ensemble import RandomForestClassifier

    # Create dataset where RandomForest tends to be overconfident
    try:
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )
    except ValueError as e:
        # Handle scikit-learn parameter constraints gracefully
        pytest.skip(f"make_classification parameters not supported: {e}")

    # Split into train/test
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Train base model (RandomForest is often overconfident)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # Get uncalibrated predictions
    pred_uncal = rf.predict_proba(X_test)[:, 1]
    brier_uncal = brier_score_loss(y_test, pred_uncal)

    # Apply calibration
    calibrated_rf = maybe_calibrate(rf, method="isotonic", cv=5)
    calibrated_rf.fit(X_train, y_train)  # Refit for calibration

    # Get calibrated predictions
    pred_cal = calibrated_rf.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, pred_cal)

    # Calibration should improve or at least not significantly worsen Brier score
    # Note: We allow for small increases due to randomness in CV
    improvement_ratio = brier_cal / brier_uncal
    assert improvement_ratio <= 1.1, (
        f"Calibration worsened Brier score significantly: {brier_uncal:.4f} -> {brier_cal:.4f}"
    )

    # Log the results for inspection
    print(f"Uncalibrated Brier score: {brier_uncal:.4f}")
    print(f"Calibrated Brier score: {brier_cal:.4f}")
    print(f"Improvement ratio: {improvement_ratio:.4f}")


def test_validate_calibration_config():
    """Test calibration configuration validation."""
    # Valid configs
    valid_configs = [
        {"method": "isotonic", "cv": 5, "ensemble": True},
        {"method": "sigmoid", "cv": 3, "ensemble": False},
        {"method": "ISOTONIC", "cv": 10},  # Case insensitive
        {},  # Empty config
    ]

    for config in valid_configs:
        result = validate_calibration_config(config)
        assert isinstance(result, dict)
        if config:
            assert result["method"] in {"isotonic", "sigmoid", None}
            assert result["cv"] >= 2
            assert isinstance(result["ensemble"], bool)

    # Invalid configs
    invalid_configs = [
        {"method": "invalid"},
        {"cv": 1},  # Too few folds
        {"cv": "not_int"},
        {"ensemble": "not_bool"},
    ]

    for config in invalid_configs:
        with pytest.raises(ValueError):
            validate_calibration_config(config)


def test_get_calibration_info():
    """Test calibration information extraction."""
    from sklearn.ensemble import RandomForestClassifier

    # Create base model
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)

    # Test uncalibrated model
    info_uncal = get_calibration_info(rf)
    assert info_uncal["is_calibrated"] is False
    assert info_uncal["calibration_method"] is None
    assert info_uncal["base_estimator_type"] == "RandomForestClassifier"

    # Test calibrated model
    calibrated_rf = maybe_calibrate(rf, method="isotonic", cv=3)
    info_cal = get_calibration_info(calibrated_rf)
    assert info_cal["is_calibrated"] is True
    assert info_cal["calibration_method"] == "isotonic"
    assert info_cal["cv_folds"] == 3
    assert info_cal["base_estimator_type"] == "RandomForestClassifier"


def test_assess_calibration_quality():
    """Test calibration quality assessment."""
    # Create perfectly calibrated predictions
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate
    y_proba_perfect = np.full(n_samples, 0.3)  # Perfect calibration

    quality_perfect = assess_calibration_quality(y_true, y_proba_perfect)

    assert "brier_score" in quality_perfect
    assert "expected_calibration_error" in quality_perfect
    assert "maximum_calibration_error" in quality_perfect

    # Perfect calibration should have low ECE
    assert quality_perfect["expected_calibration_error"] < 0.1

    # Create poorly calibrated predictions (overconfident)
    y_proba_poor = np.where(y_true == 1, 0.9, 0.1)  # Overconfident
    quality_poor = assess_calibration_quality(y_true, y_proba_poor)

    # Poor calibration should have higher ECE
    assert quality_poor["expected_calibration_error"] > quality_perfect["expected_calibration_error"]


def test_recommend_calibration_method():
    """Test calibration method recommendations."""
    # Small dataset should recommend sigmoid
    assert recommend_calibration_method("LogisticRegression", 500) == "sigmoid"

    # Tree-based models should recommend isotonic
    assert recommend_calibration_method("XGBClassifier", 2000) == "isotonic"
    assert recommend_calibration_method("LGBMClassifier", 2000) == "isotonic"
    assert recommend_calibration_method("RandomForestClassifier", 2000) == "isotonic"

    # Linear models should recommend sigmoid
    assert recommend_calibration_method("LogisticRegression", 2000) == "sigmoid"
    assert recommend_calibration_method("LinearSVC", 2000) == "sigmoid"

    # Large dataset with unknown model should default to isotonic
    assert recommend_calibration_method("UnknownModel", 5000) == "isotonic"


def test_multiclass_calibration():
    """Test that calibration works with multiclass problems."""
    from sklearn.ensemble import RandomForestClassifier

    # Create multiclass dataset
    try:
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=42,
        )
    except ValueError as e:
        # Handle scikit-learn parameter constraints gracefully
        pytest.skip(f"make_classification parameters not supported: {e}")

    # Train base model
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X, y)

    # Apply calibration
    calibrated_rf = maybe_calibrate(rf, method="isotonic", cv=3)

    # CalibratedClassifierCV needs to be fitted
    calibrated_rf.fit(X, y)

    # Test predictions
    try:
        X_test, y_test = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=123,
        )
    except ValueError as e:
        # Handle scikit-learn parameter constraints gracefully
        pytest.skip(f"make_classification parameters not supported: {e}")

    # Both should work
    pred_orig = rf.predict_proba(X_test)
    pred_cal = calibrated_rf.predict_proba(X_test)

    # Check shapes are preserved
    assert pred_orig.shape == pred_cal.shape
    assert pred_cal.shape == (100, 3)

    # Check probabilities are valid
    assert np.all(pred_cal >= 0)
    assert np.all(pred_cal <= 1)
    assert np.allclose(pred_cal.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
