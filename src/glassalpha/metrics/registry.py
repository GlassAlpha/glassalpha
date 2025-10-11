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


__all__ = ["MetricRegistry"]
