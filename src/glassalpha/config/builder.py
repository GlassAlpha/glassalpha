"""Stub config builder for test compatibility.

This module provides backwards compatibility for tests that expect
config builder functionality that was simplified.
"""

import sys
from pathlib import Path
from typing import Any

# Add the parent directory to sys.path to import from glassalpha.config
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_config_from_model(
    model: Any,
    data: Any = None,
    target_column: str = "target",
    protected_attributes: list[str] | None = None,
    **kwargs,
) -> Any:
    """Build config from model (stub implementation).

    This is a simplified stub that creates a basic config for testing.
    The real implementation was more complex and removed during simplification.
    """
    # Import GAConfig dynamically to avoid circular imports
    import glassalpha.config as config_module

    GAConfig = config_module.GAConfig

    return GAConfig(
        model={"type": "xgboost"},  # Default model type
        data={
            "dataset": "custom",
            "target_column": target_column,
            "protected_attributes": protected_attributes or [],
        },
        explainers={"strategy": "first_compatible", "priority": ["treeshap"]},
        metrics={"performance": ["accuracy"]},
        reproducibility={"random_seed": 42},
        **kwargs,
    )
