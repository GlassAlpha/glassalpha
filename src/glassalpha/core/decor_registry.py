"""Decorator-friendly registry wrapper for plugin registries.

This provides a generic solution for registries that need to support decorator
registration patterns while maintaining backward compatibility.
"""

from __future__ import annotations

from typing import Any

from .registry import PluginRegistry, PluginSpec


class DecoratorFriendlyRegistry(PluginRegistry):
    """Registry wrapper that supports decorator registration patterns.

    Supports all three registration patterns:
    1. Direct call: register("name", Class, import_check="pkg")
    2. Decorator without args: @Registry.register
    3. Decorator with args: @Registry.register("name", import_check="pkg")

    Also stores priority metadata for sorting.
    """

    def __init__(self, group: str):
        super().__init__(group)
        self._meta: dict[str, dict[str, Any]] = {}

    def register(self, name_or_obj, obj=None, **meta):
        """Register a plugin with decorator support.

        Args:
            name_or_obj: Either plugin name (str) or object to register (class/callable)
            obj: The object to register (if name string provided)
            **meta: Metadata including import_check, extra_hint, priority, etc.

        Returns:
            For decorator usage: Returns the decorator function or registered object
            For direct calls: Returns the registered object

        """
        # Direct call: register("name", Class, import_check=..., extra_hint=...)
        if obj is not None and callable(obj):
            # Create PluginSpec for optional dependency support
            spec = PluginSpec(
                name=name_or_obj,
                entry_point="",
                import_check=meta.get("import_check"),
                extra_hint=meta.get("extra_hint"),
                description=meta.get("description"),
            )
            # Register both the spec and the object
            self._specs[name_or_obj] = spec
            super().register(name_or_obj, obj)
            # Store metadata for priority, enterprise flag, and other info
            info = self._meta.setdefault(name_or_obj, {})
            info.update(
                {
                    "priority": meta.get("priority", 0),
                    "enterprise": bool(meta.get("enterprise", False)),
                },
            )
            # Keep existing metadata fields if present
            if "supports" in meta:
                info["supports"] = meta["supports"]
            return obj

        # Decorator without args: @Registry.register
        if callable(name_or_obj) and obj is None and not meta:
            cls = name_or_obj
            name = cls.__name__.lower()
            super().register(name, cls)
            self._meta.setdefault(name, {}).update({"priority": 0})
            return cls

        # Decorator with args: @Registry.register("name", import_check="pkg", ...)
        name = name_or_obj

        def deco(cls):
            # Create PluginSpec for optional dependency support
            spec = PluginSpec(
                name=name,
                entry_point="",
                import_check=meta.get("import_check"),
                extra_hint=meta.get("extra_hint"),
                description=meta.get("description"),
            )
            # Register both the spec and the object
            self._specs[name] = spec
            PluginRegistry.register(self, name, cls)
            # Store metadata for priority and other info
            self._meta.setdefault(name, {}).update(meta)
            return cls

        return deco

    def names_by_priority(self, reverse: bool = True) -> list[str]:
        """Get plugin names sorted by priority.

        Args:
            reverse: If True, sort highest priority first

        Returns:
            List of plugin names sorted by priority

        """
        self._ensure_discovered()
        return sorted(self.names(), key=lambda n: self._meta.get(n, {}).get("priority", 0), reverse=reverse)

    def get_all(self, include_enterprise: bool = True) -> list[str]:
        """Get all plugin names, optionally filtering by enterprise status.

        Args:
            include_enterprise: If True, include enterprise-only plugins.
                              If False, exclude enterprise-only plugins.

        Returns:
            List of plugin names, filtered by enterprise status

        """
        self._ensure_discovered()
        names = getattr(self, "names_by_priority", self.names)()
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

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a plugin.

        Args:
            name: Plugin name

        Returns:
            Dictionary of metadata for the plugin

        """
        return self._meta.get(name, {}).copy()
