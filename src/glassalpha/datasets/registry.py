"""Stub for test compatibility - dataset registry removed.

This module was removed during simplification but some tests still import it.
"""

from typing import NamedTuple


class DatasetSpec(NamedTuple):
    """Stub for test compatibility - dataset spec simplified."""

    name: str
    url: str = ""
    checksum: str = ""
    description: str = ""


class DatasetRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def get(name: str):
        """Get dataset by name."""
        if name == "german_credit":
            from glassalpha.datasets import load_german_credit

            return load_german_credit
        if name == "adult_income":
            from glassalpha.datasets import load_adult_income

            return load_adult_income
        raise KeyError(f"Unknown dataset: {name}")

    @staticmethod
    def list_datasets():
        """List available datasets."""
        return ["german_credit", "adult_income"]


REGISTRY = DatasetRegistry  # Alias for compatibility

__all__ = ["REGISTRY", "DatasetRegistry", "DatasetSpec"]
