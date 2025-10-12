"""Stub for test compatibility - dataset registry removed.

This module was removed during simplification but some tests still import it.
"""

from collections.abc import Callable
from typing import NamedTuple


class DatasetSpec(NamedTuple):
    """Stub for test compatibility - dataset spec simplified."""

    name: str
    url: str = ""
    checksum: str = ""
    description: str = ""
    default_relpath: str = ""  # Relative path in cache directory
    fetch_fn: Callable | None = None  # Function that returns path to dataset


class DatasetRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def get(name: str):
        """Get dataset by name - returns DatasetSpec."""
        if name == "german_credit":

            def fetch_german_credit():
                from glassalpha.datasets import load_german_credit
                from glassalpha.utils.cache_dirs import resolve_data_root

                cache_path = resolve_data_root() / "german_credit.csv"
                if not cache_path.exists():
                    # Load with encoded=True by default for sklearn compatibility
                    df = load_german_credit(encoded=True)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                return str(cache_path)

            return DatasetSpec(
                name="german_credit",
                url="built-in",  # Mark as built-in dataset
                checksum="",
                description="German Credit Risk dataset",
                default_relpath="german_credit.csv",
                fetch_fn=fetch_german_credit,
            )
        if name == "adult_income":

            def fetch_adult_income():
                from glassalpha.datasets import load_adult_income
                from glassalpha.utils.cache_dirs import resolve_data_root

                cache_path = resolve_data_root() / "adult_income.csv"
                if not cache_path.exists():
                    df = load_adult_income()
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                return str(cache_path)

            return DatasetSpec(
                name="adult_income",
                url="built-in",
                checksum="",
                description="Adult Income dataset",
                default_relpath="adult_income.csv",
                fetch_fn=fetch_adult_income,
            )
        return None  # Return None instead of raising (pipeline checks for None)

    def __getitem__(self, name: str):
        """Support subscript access: REGISTRY[name]."""
        result = self.get(name)
        if result is None:
            raise KeyError(f"Unknown dataset: {name}")
        return result

    @staticmethod
    def list_datasets():
        """List available datasets."""
        return ["german_credit", "adult_income"]


REGISTRY = DatasetRegistry()  # Create instance for compatibility

__all__ = ["REGISTRY", "DatasetRegistry", "DatasetSpec"]
