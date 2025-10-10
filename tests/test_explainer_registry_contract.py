"""Contract tests for explainer registry.

This module validates that all registered explainers implement the correct
is_compatible signature to prevent runtime TypeError issues.
"""

import pytest

from glassalpha.explain.registry import ExplainerRegistry


def test_all_explainers_have_correct_is_compatible_signature():
    """Ensure all registered explainers implement is_compatible correctly.

    This test validates the Phase 2.5 explainer API standardization by checking
    that all explainers can be called with the keyword-only signature:
        is_compatible(model=None, model_type=None, config=None)

    Failure indicates an explainer needs to be updated to match the base class signature.
    """
    # Discover all available explainers
    ExplainerRegistry.discover()

    # Get all registered explainers
    explainer_names = ExplainerRegistry.names()

    # Must have at least one explainer registered
    assert len(explainer_names) > 0, "No explainers registered - check registration"

    failures = []

    for name in explainer_names:
        try:
            # Get the explainer class
            explainer_cls = ExplainerRegistry.get(name)

            # Verify it has is_compatible method
            if not hasattr(explainer_cls, "is_compatible"):
                failures.append(f"{name}: Missing is_compatible method")
                continue

            # Try calling with keyword arguments (new standard signature)
            try:
                result = explainer_cls.is_compatible(
                    model=None,
                    model_type="test_model",
                    config={},
                )

                # Result must be a boolean
                if not isinstance(result, bool):
                    failures.append(
                        f"{name}: is_compatible returned {type(result).__name__}, expected bool",
                    )

            except TypeError as e:
                failures.append(
                    f"{name}: is_compatible has wrong signature: {e}. "
                    f"Expected: @classmethod is_compatible(cls, *, model=None, model_type=None, config=None)",
                )

        except (KeyError, ImportError) as e:
            # Explainer not available (e.g., missing dependencies)
            # This is OK - just skip it
            pytest.skip(f"Explainer {name} not available: {e}")

        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: Unexpected error: {e}")

    # Report all failures at once for better debugging
    if failures:
        pytest.fail(
            "Explainer contract violations found:\n  - " + "\n  - ".join(failures),
        )


def test_explainer_is_compatible_with_various_inputs():
    """Test that is_compatible handles different input combinations correctly."""
    ExplainerRegistry.discover()

    for name in ExplainerRegistry.names():
        try:
            explainer_cls = ExplainerRegistry.get(name)

            # Test 1: No arguments (all optional)
            result1 = explainer_cls.is_compatible()
            assert isinstance(result1, bool), f"{name}: Result should be bool with no args"

            # Test 2: Only model_type
            result2 = explainer_cls.is_compatible(model_type="xgboost")
            assert isinstance(result2, bool), f"{name}: Result should be bool with model_type only"

            # Test 3: Only model (string)
            result3 = explainer_cls.is_compatible(model="lightgbm")
            assert isinstance(result3, bool), f"{name}: Result should be bool with model only"

            # Test 4: Both model and model_type
            result4 = explainer_cls.is_compatible(model="xgboost", model_type="xgboost")
            assert isinstance(result4, bool), f"{name}: Result should be bool with both args"

            # Test 5: With config
            result5 = explainer_cls.is_compatible(model_type="xgboost", config={"test": True})
            assert isinstance(result5, bool), f"{name}: Result should be bool with config"

        except (KeyError, ImportError):
            # Skip unavailable explainers
            pytest.skip(f"Explainer {name} not available")
        except TypeError as e:
            pytest.fail(f"{name}: is_compatible failed with TypeError: {e}")


def test_registry_is_compatible_method():
    """Test that registry's is_compatible method works with new explainer signatures."""
    ExplainerRegistry.discover()

    # Test with a known explainer (should work if any are available)
    names = ExplainerRegistry.names()
    if not names:
        pytest.skip("No explainers available")

    # Pick first available explainer
    test_name = names[0]

    # Test registry's is_compatible wrapper
    result = ExplainerRegistry.is_compatible(test_name, model_type="test", model=None)
    assert isinstance(result, bool), "Registry is_compatible should return bool"


def test_noop_explainer_always_compatible():
    """Verify NoOpExplainer is always compatible (fallback behavior)."""
    ExplainerRegistry.discover()

    if not ExplainerRegistry.has("noop"):
        pytest.skip("NoOpExplainer not available")

    # NoOp should be compatible with everything
    assert ExplainerRegistry.is_compatible("noop", model_type="anything")
    assert ExplainerRegistry.is_compatible("noop", model_type="test", model=None)
