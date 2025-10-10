"""Test decorator-friendly registry functionality."""

from glassalpha.core.decor_registry import DecoratorFriendlyRegistry


class MockPlugin:
    """Mock plugin for testing."""

    def __init__(self, name="mock"):
        self.name = name


def test_decorator_friendly_registry_direct_registration():
    """Test direct registration call."""
    registry = DecoratorFriendlyRegistry("test.group")

    class DirectPlugin:
        pass

    result = registry.register("direct_plugin", DirectPlugin, priority=50)

    assert result is DirectPlugin
    assert "direct_plugin" in registry.names()
    assert registry._meta["direct_plugin"]["priority"] == 50


def test_decorator_friendly_registry_decorator_without_args():
    """Test decorator registration without arguments."""
    registry = DecoratorFriendlyRegistry("test.group")

    @registry.register
    class AutoNamedPlugin:
        pass

    # Should be registered with lowercase class name
    assert "autonamedplugin" in registry.names()
    assert registry._meta["autonamedplugin"]["priority"] == 0


def test_decorator_friendly_registry_decorator_with_args():
    """Test decorator registration with arguments."""
    registry = DecoratorFriendlyRegistry("test.group")

    @registry.register("custom_plugin", priority=25, import_check="test_dep")
    class CustomPlugin:
        pass

    # Should be registered with explicit name
    assert "custom_plugin" in registry.names()
    assert registry._meta["custom_plugin"]["priority"] == 25


def test_decorator_friendly_registry_priority_sorting():
    """Test that priorities are used for sorting."""
    registry = DecoratorFriendlyRegistry("test.group")

    @registry.register("low_priority", priority=10)
    class LowPriorityPlugin:
        pass

    @registry.register("high_priority", priority=100)
    class HighPriorityPlugin:
        pass

    @registry.register("medium_priority", priority=50)
    class MediumPriorityPlugin:
        pass

    # Test sorting by priority (highest first by default)
    sorted_names = registry.names_by_priority(reverse=True)
    assert sorted_names[0] == "high_priority"
    assert sorted_names[1] == "medium_priority"
    assert sorted_names[2] == "low_priority"

    # Test sorting by priority (lowest first)
    sorted_names = registry.names_by_priority(reverse=False)
    assert sorted_names[0] == "low_priority"
    assert sorted_names[1] == "medium_priority"
    assert sorted_names[2] == "high_priority"


def test_decorator_friendly_registry_metadata_storage():
    """Test that metadata is properly stored and accessible."""
    registry = DecoratorFriendlyRegistry("test.group")

    @registry.register("test_plugin", priority=75, import_check="test_lib", extra_hint="test_hint")
    class TestPlugin:
        pass

    # Check that metadata is stored
    assert "test_plugin" in registry._meta
    meta = registry._meta["test_plugin"]
    assert meta["priority"] == 75
    assert meta["import_check"] == "test_lib"
    assert meta["extra_hint"] == "test_hint"


def test_decorator_friendly_registry_plugin_spec_integration():
    """Test integration with PluginSpec for optional dependencies."""
    registry = DecoratorFriendlyRegistry("test.group")

    # Test that PluginSpec objects are created correctly
    @registry.register("spec_plugin", import_check="optional_lib", extra_hint="optional")
    class SpecPlugin:
        pass

    # Should have created a PluginSpec
    assert "spec_plugin" in registry._specs
    spec = registry._specs["spec_plugin"]
    assert spec.import_check == "optional_lib"
    assert spec.extra_hint == "optional"
