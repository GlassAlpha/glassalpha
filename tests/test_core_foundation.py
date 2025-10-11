"""Test core foundation components work together.

This tests the basic architecture patterns: interfaces, registries,
and NoOp implementations.

NOTE: These tests rely on the removed registry system and are expected to fail
after simplification. They are kept for backwards compatibility but marked
with @pytest.mark.registry_removed.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from glassalpha.core import (
    ModelRegistry,
    NoOpMetric,
    PassThroughModel,
    list_components,
    select_explainer,
)

# Mark all tests in this file as registry_removed
pytestmark = pytest.mark.registry_removed


def test_passthrough_model_works():
    """Test PassThrough model generates predictions."""
    model = PassThroughModel(default_value=0.7)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    predictions = model.predict(df)
    assert len(predictions) == 3
    assert all(p == 0.7 for p in predictions)

    proba = model.predict_proba(df)
    assert proba.shape == (3, 2)
    assert all(p[1] == 0.7 for p in proba)


def test_noop_explainer_works():
    """Test NoOp explainer returns valid structure."""
    # Test that noop explainer exists and is registered
    from glassalpha.explain import select_explainer  # ExplainerRegistry

    explainer_class = ExplainerRegistry.get("noop")
    assert explainer_class is not None
    model = PassThroughModel()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    explanation = explainer_class().explain(model, df)

    assert explanation["status"] == "no_explanation"
    assert explanation["shap_values"].shape == (2, 2)
    assert explainer_class().supports_model(model)


def test_noop_metric_works():
    """Test NoOp metric returns placeholder values."""
    metric = NoOpMetric()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = metric.compute(y_true, y_pred)

    assert "noop_metric" in result
    assert result["samples_processed"] == 4
    assert not metric.requires_sensitive_features()


def test_registry_registration():
    """Test components are registered via entry points."""
    # Check PassThrough model is registered via entry point
    model_cls = ModelRegistry.get("passthrough")
    assert model_cls is not None
    assert model_cls == PassThroughModel


def test_deterministic_explainer_selection():
    """Test explainer selection is deterministic."""
    config1 = {"explainers": {"priority": ["noop", "nonexistent"]}}
    config2 = {"explainers": {"priority": ["noop", "nonexistent"]}}  # Same order

    selected1 = select_explainer("xgboost", config1)
    selected2 = select_explainer("xgboost", config2)

    # Should select noop explainer when explicitly requested in priority
    assert selected1 == selected2 == "noop"


def test_list_components():
    """Test listing registered components."""
    # Import models to trigger registration

    components = list_components()

    assert "models" in components
    assert "passthrough" in components["models"]


def test_registry_priority_selection():
    """Test priority-based selection with fallback."""
    # Simplified test - just check that priority selection works
    config = {"explainers": {"priority": ["noop", "nonexistent"]}}

    # Should select first in priority list
    selected = select_explainer("test_model", config)
    assert selected == "noop"

    # Should select noop explainer for unknown model since it's in priority list
    selected = select_explainer("unknown_model", config)
    assert selected == "noop"  # NoOp is compatible with everything when explicitly requested
