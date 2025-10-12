"""Stub for test compatibility - metrics registry removed.

This module was removed during simplification but some tests still import it.
"""


class MetricRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def discover():
        pass  # No-op for compatibility

    @staticmethod
    def available_plugins():
        return {
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1": True,
            "auc_roc": True,
        }

    @classmethod
    def get(cls, name: str):
        """Get metric class by name (stub)."""
        # Return None for compatibility
        return

    @classmethod
    def get_all(cls):
        """Get all registered metrics (stub)."""
        return {}


__all__ = ["MetricRegistry"]
