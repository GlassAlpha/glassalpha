"""Data loading and preprocessing modules.

This package provides data loaders for different data modalities
with schema validation, protected attributes handling, and
reproducible preprocessing.
"""

from .base import DataInterface, DataSchema
from .tabular import (
    TabularDataLoader,
    TabularDataSchema,
    create_schema_from_data,
    load_tabular_data,
)

__all__ = [
    "DataInterface",
    "DataSchema",
    "TabularDataLoader",
    "TabularDataSchema",
    "create_schema_from_data",
    "load_tabular_data",
]
