from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginSpec:
    """Specification for a plugin with dependency information."""

    name: str
    entry_point: str  # "module.path:function_function"
    import_check: str | None  # Package name to check (e.g., "xgboost")
    extra_hint: str | None  # Optional dependency group name (e.g., "xgboost")
    description: str | None = None


class PluginRegistry:
    """Simple entry-point aware registry with lazy loading."""

    def __init__(self, group: str):
        self.group = group
        self._entry_points: dict[str, str] = {}  # name -> "module:attr"
        self._objects: dict[str, Any] = {}  # name -> loaded object (class/callable)
        self._specs: dict[str, PluginSpec] = {}  # name -> PluginSpec
        self._meta: dict[str, dict[str, Any]] = {}  # name -> metadata
        self._aliases: dict[str, str] = {}  # alias -> canonical_name
        self._discovered = False

    # ----- public API expected by tests -----

    def register(self, name_or_obj, obj=None, **meta):
        """Register a plugin with decorator support.

        Args:
            name_or_obj: Either plugin name (str) or object to register (class/callable)
            obj: The object to register (if name string provided)
            **meta: Metadata including import_check, extra_hint, priority, etc.

        Returns:
            For decorator usage: Returns the decorator function or registered object
            For direct calls: Returns None

        """
        # Direct call: register("name", Class, import_check=..., extra_hint=...)
        if obj is not None and callable(obj):
            # Register object directly
            name = name_or_obj
            self._objects[name] = obj
            # Store metadata for priority, enterprise flag, and other info
            info = self._meta.setdefault(name, {})
            info.update(
                {
                    "priority": meta.get("priority", 0),
                    "enterprise": bool(meta.get("enterprise", False)),
                },
            )
            logger.debug(f"Registered plugin '{name}' with object")
            return obj

        # Decorator without args: @Registry.register
        if callable(name_or_obj) and obj is None and not meta:
            cls = name_or_obj
            name = cls.__name__.lower()
            self._objects[name] = cls
            self._meta.setdefault(name, {}).update({"priority": 0})
            logger.debug(f"Registered plugin '{name}' with class")
            return cls

        # Decorator with args: @Registry.register("name", import_check="pkg", ...)
        name = name_or_obj

        def deco(cls):
            # Register both the spec and the object
            if meta.get("import_check") or meta.get("extra_hint"):
                spec = PluginSpec(
                    name=name,
                    entry_point="",
                    import_check=meta.get("import_check"),
                    extra_hint=meta.get("extra_hint"),
                    description=meta.get("description"),
                )
                self._specs[name] = spec

            self._objects[name] = cls
            # Store metadata for priority, enterprise flag, and other info
            info = self._meta.setdefault(name, {})
            info.update(
                {
                    "priority": meta.get("priority", 0),
                    "enterprise": bool(meta.get("enterprise", False)),
                },
            )
            logger.debug(f"Registered plugin '{name}' with class via decorator")
            return cls

        return deco

    def alias(self, alias: str, target: str) -> None:
        """Create an alias for a plugin name.

        Args:
            alias: The alias name to create
            target: The canonical name the alias points to

        """
        self._aliases[alias] = target

    def get(self, name: str) -> Any:
        """Return the registered object (class/callable). Lazy-load from entry points if needed."""
        self._ensure_discovered()

        # Resolve alias if present
        canonical_name = self._aliases.get(name, name)

        if canonical_name in self._objects:
            return self._objects[canonical_name]
        if canonical_name in self._entry_points:
            module, attr = self._entry_points[canonical_name].split(":")
            obj = getattr(__import__(module, fromlist=[attr]), attr)
            self._objects[canonical_name] = obj
            return obj
        if canonical_name in self._specs:
            # For backward compatibility, also check specs
            return self._specs[canonical_name]
        raise KeyError(f"Unknown plugin '{name}' for group '{self.group}'")

    # ----- convenience, not required by the failing test -----

    def has(self, name: str) -> bool:
        self._ensure_discovered()
        canonical_name = self._aliases.get(name, name)
        return canonical_name in self._objects or canonical_name in self._entry_points

    def names(self) -> list[str]:
        self._ensure_discovered()
        names = set(self._objects) | set(self._entry_points) | set(self._aliases.keys())
        return sorted(names)

    def primary_names(self) -> list[str]:
        """Get list of primary plugin names (excluding aliases).

        Returns:
            Sorted list of primary plugin names (aliases are excluded)

        """
        self._ensure_discovered()
        # Only return actual registered names, not aliases
        names = set(self._objects) | set(self._entry_points)
        return sorted(names)

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a plugin.

        Args:
            name: Plugin name

        Returns:
            Dictionary of metadata for the plugin

        """
        return self._meta.get(name, {}).copy()

    def available_plugins(self) -> dict[str, bool]:
        """Get availability status of all plugins.

        Returns:
            Dictionary mapping plugin names to availability status

        """
        availability = {}
        all_names = set(self.names()) | set(self._specs.keys())

        for name in all_names:
            # Check if plugin has dependency requirements
            if name in self._specs:
                spec = self._specs[name]
                if spec.import_check and find_spec(spec.import_check) is None:
                    availability[name] = False
                else:
                    availability[name] = True
            else:
                # For directly registered objects, assume available
                availability[name] = True

        return availability

    def get_install_hint(self, name: str) -> str | None:
        """Get installation hint for a plugin.

        Args:
            name: Plugin name

        Returns:
            Installation hint string or None if no hint available

        """
        # Simple implementation - could be enhanced
        if name in ["xgboost", "lightgbm"]:
            return f"pip install 'glassalpha[{name}]'"
        return None

    def get_all(self, include_enterprise: bool = True) -> list[str]:
        """Get all plugin names, optionally filtering by enterprise status.

        Args:
            include_enterprise: If True, include enterprise-only plugins.
                              If False, exclude enterprise-only plugins.

        Returns:
            List of plugin names, filtered by enterprise status

        """
        self._ensure_discovered()
        names = self.names()
        if include_enterprise:
            return names
        return [n for n in names if not (self._meta.get(n, {}).get("enterprise", False))]

    def is_enterprise(self, name: str) -> bool:
        """Check if a plugin is enterprise-only.

        Args:
            name: Plugin name

        Returns:
            True if plugin is enterprise-only, False otherwise

        """
        return self._meta.get(name, {}).get("enterprise", False)

    def load(self, name: str, *args, **kwargs) -> Any:
        """Load a plugin instance.

        Args:
            name: Plugin name
            **kwargs: Arguments to pass to plugin constructor

        Returns:
            Plugin instance

        Raises:
            KeyError: If plugin not found
            ImportError: If optional dependency missing with installation hint

        """
        # Check if plugin is available before loading
        available = self.available_plugins()
        if not available.get(name, False):
            hint = self.get_install_hint(name)
            if hint:
                raise ImportError(
                    f"Missing optional dependency for plugin '{name}'. {hint}",
                )
            raise ImportError(f"Plugin '{name}' is not available.")

        obj = self.get(name)
        try:
            return obj(*args, **kwargs)
        except TypeError:
            return obj

    # ----- entry point discovery -----

    def discover(self) -> None:
        """Populate entry points but do not import heavy modules yet."""
        self._entry_points.clear()
        for ep in metadata.entry_points(group=self.group):
            self._entry_points[ep.name] = ep.value
        self._discovered = True

    # ----- internals -----

    def _ensure_discovered(self) -> None:
        if not self._discovered:
            self.discover()


# Global registries for different component types
# ModelRegistry and DataRegistry are defined here because they don't need decorator support
ModelRegistry = PluginRegistry("glassalpha.models")
DataRegistry = PluginRegistry("glassalpha.data_handlers")

# Discover plugins on import
ModelRegistry.discover()
DataRegistry.discover()

# Import model modules to register them
# This ensures models are available when ModelRegistry is used
# The imports trigger model registration via @ModelRegistry.register decorators
try:
    from ..models.tabular import lightgbm, sklearn, xgboost  # noqa: F401
except ImportError:
    # Optional dependencies - models will be unavailable if not installed
    pass

# Note: ExplainerRegistry, MetricRegistry, and ProfileRegistry are defined in their
# respective modules (explain/registry.py, metrics/registry.py, profiles/registry.py)
# and re-exported through core/__init__.py for convenience


def list_components(component_type: str | None = None, include_enterprise: bool = False) -> dict[str, list[str]]:
    """List all registered components.

    Args:
        component_type: Specific type to list, or None for all
        include_enterprise: Whether to include enterprise components

    Returns:
        Dictionary mapping component types to lists of names

    """
    # Import registries from their canonical locations
    from ..explain.registry import ExplainerRegistry
    from ..metrics.registry import MetricRegistry
    from ..profiles.registry import ProfileRegistry

    registries = {
        "models": ModelRegistry,
        "explainers": ExplainerRegistry,
        "metrics": MetricRegistry,
        "data_handlers": DataRegistry,
        "profiles": ProfileRegistry,
    }

    if component_type:
        registry = registries.get(component_type)
        if registry:
            # Use primary_names to exclude aliases for cleaner output
            names = registry.primary_names() if hasattr(registry, "primary_names") else registry.names()
            return {component_type: names}
        return {}

    result = {}
    for name, registry in registries.items():
        # Use primary_names to exclude aliases for cleaner output
        names = registry.primary_names() if hasattr(registry, "primary_names") else registry.names()
        result[name] = names

    return result


def select_explainer(model_type: str, config: dict[str, Any]) -> str | None:
    """Select appropriate explainer based on model and config.

    This ensures deterministic selection based on configuration priorities.

    Args:
        model_type: Type of model being explained
        config: Configuration with explainer priorities

    Returns:
        Selected explainer name or None

    """
    # Import from canonical location
    from ..explain.registry import ExplainerRegistry

    try:
        return ExplainerRegistry.find_compatible(model_type, config)
    except RuntimeError:
        return None


def instantiate_explainer(name: str, **kwargs: Any) -> Any:
    """Instantiate an explainer by name.

    Args:
        name: Explainer name to instantiate
        **kwargs: Arguments to pass to explainer constructor

    Returns:
        Instantiated explainer object

    Raises:
        KeyError: If explainer name not found
        ImportError: If explainer dependencies not available

    """
    # Import from canonical location
    from ..explain.registry import ExplainerRegistry

    cls = ExplainerRegistry.get(name)
    return cls(**kwargs)
