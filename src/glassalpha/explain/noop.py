"""No-op explainer for testing and fallback scenarios."""

from __future__ import annotations

from glassalpha.explain.base import ExplainerBase


class NoOpExplainer(ExplainerBase):
    """No-op explainer that does nothing.

    Used for testing and as a fallback when no other explainers are available.
    """

    name = "noop"
    priority = 0  # Lowest priority
    version = "1.0.0"

    def __init__(self) -> None:
        """Initialize no-op explainer."""
        super().__init__()

    def explain(self, X, y=None, model=None):
        """Return empty explanation (no-op)."""
        return {"explanation": "noop", "status": "no_operation"}

    def is_available(self) -> bool:
        """No-op explainer is always available."""
        return True


# Global instance for registration
noop = NoOpExplainer()
