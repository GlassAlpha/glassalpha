"""Dataset loaders for common benchmark datasets.

This package provides loaders for canonical machine learning datasets
commonly used for fairness and compliance auditing.

Performance note: This module uses lazy imports via __getattr__ to avoid
loading heavy dependencies (pandas, numpy) during CLI --help rendering.
Dataset loaders are imported only when actually accessed, but registration
happens immediately to populate REGISTRY.
"""

from typing import Any

# Register datasets immediately (lightweight operation)
# This populates REGISTRY but doesn't import heavy data loaders
from .register_builtin import register_builtin_datasets

# Import registry - this is lightweight
from .registry import REGISTRY, DatasetSpec

register_builtin_datasets()

__all__ = [
    "REGISTRY",
    "AdultIncomeDataset",
    "DatasetSpec",
    "GermanCreditDataset",
    "get_adult_income_schema",
    "get_german_credit_schema",
    "load_adult_income",
    "load_german_credit",
]


def __getattr__(name: str) -> Any:
    """Lazy import dataset loaders to avoid heavy imports during CLI --help.

    This function is called when an attribute is not found in the module,
    allowing us to defer expensive imports until they're actually needed.

    Registration happens at module import time, but the heavy data loaders
    (german_credit, adult_income modules with pandas/numpy) are imported lazily.
    """
    # Import adult_income components on demand
    if name == "AdultIncomeDataset":
        from .adult_income import AdultIncomeDataset

        return AdultIncomeDataset
    if name == "get_adult_income_schema":
        from .adult_income import get_adult_income_schema

        return get_adult_income_schema
    if name == "load_adult_income":
        from .adult_income import load_adult_income

        return load_adult_income

    # Import german_credit components on demand
    if name == "GermanCreditDataset":
        from .german_credit import GermanCreditDataset

        return GermanCreditDataset
    if name == "get_german_credit_schema":
        from .german_credit import get_german_credit_schema

        return get_german_credit_schema
    if name == "load_german_credit":
        from .german_credit import load_german_credit

        return load_german_credit

    # If not found, raise AttributeError as expected
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
