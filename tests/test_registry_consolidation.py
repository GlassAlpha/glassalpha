"""Tests for registry consolidation - verifying clean imports and no proxies."""

import sys


def test_all_registries_importable():
    """Verify all registries can be imported from their canonical locations."""
    from glassalpha.core import (
        DataRegistry,
        ExplainerRegistry,
        MetricRegistry,
        ModelRegistry,
        ProfileRegistry,
    )

    assert ModelRegistry is not None
    assert ExplainerRegistry is not None
    assert MetricRegistry is not None
    assert ProfileRegistry is not None
    assert DataRegistry is not None


def test_registries_from_canonical_locations():
    """Verify registries can be imported directly from their canonical locations."""
    from glassalpha.core.registry import DataRegistry, ModelRegistry
    from glassalpha.explain.registry import ExplainerRegistry
    from glassalpha.metrics.registry import MetricRegistry
    from glassalpha.profiles.registry import ProfileRegistry

    assert ModelRegistry is not None
    assert ExplainerRegistry is not None
    assert MetricRegistry is not None
    assert ProfileRegistry is not None
    assert DataRegistry is not None


def test_registry_re_exports_are_same_instance():
    """Verify that re-exports are the same instances as canonical imports."""
    from glassalpha.core import ExplainerRegistry as CoreExplainerRegistry
    from glassalpha.core import MetricRegistry as CoreMetricRegistry
    from glassalpha.core import ProfileRegistry as CoreProfileRegistry
    from glassalpha.explain.registry import ExplainerRegistry
    from glassalpha.metrics.registry import MetricRegistry
    from glassalpha.profiles.registry import ProfileRegistry

    # Should be the exact same instances (not proxies)
    assert CoreExplainerRegistry is ExplainerRegistry
    assert CoreMetricRegistry is MetricRegistry
    assert CoreProfileRegistry is ProfileRegistry


def test_registry_basic_operations():
    """Test basic registry operations work."""
    from glassalpha.core import ModelRegistry

    # Test registration
    test_model_class = type("TestModel", (), {"predict": lambda self, X: X})
    ModelRegistry.register("test_model_temp", test_model_class)
    assert ModelRegistry.has("test_model_temp")

    # Test retrieval
    model = ModelRegistry.get("test_model_temp")
    assert model == test_model_class

    # Test listing
    assert "test_model_temp" in ModelRegistry.names()


def test_no_circular_imports():
    """Verify no circular import issues when importing in various orders."""
    # Clear module cache
    modules_to_clear = [k for k in sys.modules if k.startswith("glassalpha")]
    original_modules = {k: sys.modules[k] for k in modules_to_clear}

    try:
        # Clear
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import in one order
        from glassalpha.core import ModelRegistry

        assert ModelRegistry is not None

        # Clear again
        modules_to_clear = [k for k in sys.modules if k.startswith("glassalpha")]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import in another order
        from glassalpha.explain.registry import ExplainerRegistry
        from glassalpha.metrics.registry import MetricRegistry

        assert ExplainerRegistry is not None
        assert MetricRegistry is not None

    finally:
        # Restore original modules
        for k, v in original_modules.items():
            sys.modules[k] = v


def test_decorator_registration_still_works():
    """Test decorator-based registration still works with metrics."""
    from glassalpha.metrics.registry import MetricRegistry

    @MetricRegistry.register("test_consolidation_metric")
    class TestMetric:
        pass

    assert MetricRegistry.has("test_consolidation_metric")
    assert MetricRegistry.get("test_consolidation_metric") == TestMetric


def test_no_proxy_classes_exist():
    """Verify no proxy classes exist in the registry module."""
    import inspect

    from glassalpha.core import registry

    # Get all classes defined in the registry module
    classes = [
        name
        for name, obj in inspect.getmembers(registry, inspect.isclass)
        if obj.__module__ == "glassalpha.core.registry"
    ]

    # Should not have any Proxy classes
    proxy_classes = [c for c in classes if "Proxy" in c]
    assert len(proxy_classes) == 0, f"Found proxy classes: {proxy_classes}"


def test_list_components_works():
    """Test list_components utility function works."""
    from glassalpha.core.registry import list_components

    # Test listing all components
    all_components = list_components()
    assert "models" in all_components
    assert "explainers" in all_components
    assert "metrics" in all_components
    assert "profiles" in all_components

    # Test listing specific component type
    models = list_components(component_type="models")
    assert "models" in models
    assert isinstance(models["models"], list)


def test_select_explainer_works():
    """Test select_explainer utility function works."""
    from glassalpha.core.registry import select_explainer

    config = {"explainers": {"priority": ["noop"]}}

    # Should be able to select explainer
    result = select_explainer("xgboost", config)
    # Result can be a string or None depending on availability
    assert result is None or isinstance(result, str)


def test_instantiate_explainer_works():
    """Test instantiate_explainer utility function works."""
    from glassalpha.core.registry import instantiate_explainer

    # Should be able to instantiate noop explainer
    explainer = instantiate_explainer("noop")
    assert explainer is not None


def test_registries_have_discover_method():
    """Verify all registries have discover method."""
    from glassalpha.core import (
        DataRegistry,
        ExplainerRegistry,
        MetricRegistry,
        ModelRegistry,
        ProfileRegistry,
    )

    assert hasattr(ModelRegistry, "discover")
    assert hasattr(ExplainerRegistry, "discover")
    assert hasattr(MetricRegistry, "discover")
    assert hasattr(ProfileRegistry, "discover")
    assert hasattr(DataRegistry, "discover")


def test_registries_have_names_method():
    """Verify all registries have names method."""
    from glassalpha.core import (
        DataRegistry,
        ExplainerRegistry,
        MetricRegistry,
        ModelRegistry,
        ProfileRegistry,
    )

    assert hasattr(ModelRegistry, "names")
    assert hasattr(ExplainerRegistry, "names")
    assert hasattr(MetricRegistry, "names")
    assert hasattr(ProfileRegistry, "names")
    assert hasattr(DataRegistry, "names")

    # Verify they return lists
    assert isinstance(ModelRegistry.names(), list)
    assert isinstance(ExplainerRegistry.names(), list)
    assert isinstance(MetricRegistry.names(), list)
    assert isinstance(ProfileRegistry.names(), list)
    assert isinstance(DataRegistry.names(), list)
