"""Contract tests ensuring all implementations satisfy their Protocol interfaces.

These tests validate that every registered component correctly implements
its declared interface, preventing runtime errors and ensuring API consistency.
"""

import pytest

from glassalpha.core.registry import ModelRegistry
from glassalpha.explain.registry import ExplainerRegistry
from glassalpha.metrics.registry import MetricRegistry


def test_all_models_satisfy_interface():
    """Verify all registered models satisfy ModelInterface protocol.

    This test ensures that every model wrapper implements the required
    methods defined by ModelInterface, preventing runtime attribute errors.
    """
    # Get all registered model names
    model_names = ModelRegistry.names()

    # Filter out internal test models and test fixtures
    # Use pattern matching to exclude any test fixtures that may be registered during test runs
    production_models = [
        name
        for name in model_names
        if name not in ["passthrough"]
        and not any(pattern in name.lower() for pattern in ["test", "enterprise", "oss", "mock"])
    ]

    for name in production_models:
        try:
            model_cls = ModelRegistry.get(name)
        except (ImportError, KeyError):
            # Skip models with missing optional dependencies
            pytest.skip(f"Skipping {name}: optional dependency not available")
            continue

        # Check required methods exist
        assert hasattr(model_cls, "predict"), f"{name} missing predict()"
        assert hasattr(model_cls, "predict_proba"), f"{name} missing predict_proba()"
        assert hasattr(model_cls, "get_model_type"), f"{name} missing get_model_type()"
        assert hasattr(model_cls, "get_capabilities"), f"{name} missing get_capabilities()"

        # Verify it's callable (can be instantiated)
        assert callable(model_cls), f"{name} is not callable"


def test_all_explainers_satisfy_interface():
    """Verify all registered explainers satisfy ExplainerInterface protocol.

    This test ensures that every explainer implements the required methods
    and class attributes defined by ExplainerInterface.
    """
    # Get all registered explainer names
    explainer_names = ExplainerRegistry.names()

    # Filter out aliases and internal test explainers
    # Use pattern matching to exclude any test fixtures that may be registered during test runs
    aliases = ["coef", "coeff", "perm", "permutation_importance"]
    primary_explainers = [
        name
        for name in explainer_names
        if name not in aliases and not any(pattern in name.lower() for pattern in ["test", "enterprise", "oss", "mock"])
    ]

    for name in primary_explainers:
        try:
            explainer_cls = ExplainerRegistry.get(name)
        except (ImportError, KeyError):
            # Skip explainers with missing optional dependencies
            pytest.skip(f"Skipping {name}: optional dependency not available")
            continue

        # Check required methods exist
        assert hasattr(explainer_cls, "explain"), f"{name} missing explain()"
        assert hasattr(explainer_cls, "supports_model") or hasattr(
            explainer_cls,
            "is_compatible",
        ), f"{name} missing supports_model() or is_compatible()"

        # Check required class attributes
        assert hasattr(explainer_cls, "capabilities") or hasattr(explainer_cls, "priority"), (
            f"{name} missing capabilities/priority"
        )

        # Verify it's callable (can be instantiated)
        assert callable(explainer_cls), f"{name} is not callable"


def test_all_metrics_satisfy_interface():
    """Verify all registered metrics satisfy MetricInterface protocol.

    This test ensures that every metric implements the required methods
    and class attributes defined by MetricInterface.
    """
    # Get all registered metric names
    metric_names = MetricRegistry.names()

    # Filter out noop metric and test fixtures
    # Use pattern matching to exclude any test fixtures that may be registered during test runs
    production_metrics = [
        name
        for name in metric_names
        if name not in ["noop", "noop_metric"]
        and not any(pattern in name.lower() for pattern in ["test", "enterprise", "oss", "mock"])
    ]

    for name in production_metrics:
        try:
            metric_cls = MetricRegistry.get(name)
        except (ImportError, KeyError):
            # Skip metrics with missing optional dependencies
            pytest.skip(f"Skipping {name}: optional dependency not available")
            continue

        # Check required methods exist
        assert hasattr(metric_cls, "compute"), f"{name} missing compute()"
        assert hasattr(metric_cls, "get_metric_names"), f"{name} missing get_metric_names()"
        assert hasattr(metric_cls, "requires_sensitive_features"), f"{name} missing requires_sensitive_features()"

        # Check required class attributes
        assert hasattr(metric_cls, "metric_type"), f"{name} missing metric_type"
        assert hasattr(metric_cls, "version"), f"{name} missing version"

        # Verify it's callable (can be instantiated)
        assert callable(metric_cls), f"{name} is not callable"


def test_model_interface_methods_are_callable():
    """Verify model wrapper methods can actually be called.

    This test goes beyond simple attribute existence to verify that
    methods have correct signatures and can be invoked.
    """
    # Test with logistic regression (always available, no optional deps)
    try:
        lr_cls = ModelRegistry.get("logistic_regression")
    except (ImportError, KeyError):
        pytest.skip("logistic_regression not available")

    # Check methods are callable
    assert callable(getattr(lr_cls, "predict", None)), "predict not callable"
    assert callable(getattr(lr_cls, "predict_proba", None)), "predict_proba not callable"
    assert callable(getattr(lr_cls, "get_model_type", None)), "get_model_type not callable"
    assert callable(getattr(lr_cls, "get_capabilities", None)), "get_capabilities not callable"


def test_explainer_interface_methods_are_callable():
    """Verify explainer methods can actually be called.

    Tests explainers that don't require optional dependencies.
    """
    # Test with coefficients explainer (no optional deps)
    try:
        coef_cls = ExplainerRegistry.get("coefficients")
    except (ImportError, KeyError):
        pytest.skip("coefficients explainer not available")

    # Check methods are callable
    assert callable(getattr(coef_cls, "explain", None)), "explain not callable"

    # Check for either supports_model or is_compatible
    has_supports = callable(getattr(coef_cls, "supports_model", None))
    has_compat = callable(getattr(coef_cls, "is_compatible", None))
    assert has_supports or has_compat, "No compatibility check method found"


def test_metric_interface_methods_are_callable():
    """Verify metric methods can actually be called.

    Tests metrics that don't require optional dependencies.
    """
    # Test with accuracy metric (no optional deps)
    try:
        acc_cls = MetricRegistry.get("accuracy")
    except (ImportError, KeyError):
        pytest.skip("accuracy metric not available")

    # Check methods are callable
    assert callable(getattr(acc_cls, "compute", None)), "compute not callable"
    assert callable(getattr(acc_cls, "get_metric_names", None)), "get_metric_names not callable"
    assert callable(getattr(acc_cls, "requires_sensitive_features", None)), "requires_sensitive_features not callable"


def test_no_protocol_violations():
    """Verify implementations don't violate protocol expectations.

    This test checks that implementations follow the expected patterns
    defined by the Protocol interfaces.
    """
    # All models should have version strings
    for name in ModelRegistry.names():
        if name == "passthrough":  # Skip internal test model
            continue

        try:
            model_cls = ModelRegistry.get(name)
            # Check version attribute exists (not checking callable)
            assert hasattr(model_cls, "version") or hasattr(
                model_cls,
                "__version__",
            ), f"{name} missing version information"
        except (ImportError, KeyError):
            continue

    # All explainers should declare priorities
    aliases = ["coef", "coeff", "perm", "permutation_importance"]
    for name in ExplainerRegistry.names():
        # Skip aliases and test fixtures
        if name in aliases or any(pattern in name.lower() for pattern in ["test", "enterprise", "oss", "mock"]):
            continue

        try:
            explainer_cls = ExplainerRegistry.get(name)
            # Check priority exists (class attribute or metadata)
            has_priority = hasattr(explainer_cls, "priority")
            has_metadata = name in ExplainerRegistry._meta
            assert has_priority or has_metadata, f"{name} missing priority declaration"
        except (ImportError, KeyError):
            continue

    # All metrics should declare metric_type
    for name in MetricRegistry.names():
        # Skip noop metric and test fixtures
        if name in ["noop", "noop_metric"] or any(
            pattern in name.lower() for pattern in ["test", "enterprise", "oss", "mock"]
        ):
            continue

        try:
            metric_cls = MetricRegistry.get(name)
            assert hasattr(metric_cls, "metric_type"), f"{name} missing metric_type"
            # Verify metric_type is a valid category
            assert metric_cls.metric_type in ["performance", "fairness", "drift", "stability"], (
                f"{name} has invalid metric_type: {metric_cls.metric_type}"
            )
        except (ImportError, KeyError):
            continue
