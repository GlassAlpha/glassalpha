"""Dataset loaders for common benchmark datasets.

This package provides loaders for canonical machine learning datasets
commonly used for fairness and compliance auditing.

Performance note: This module uses lazy imports via __getattr__ to avoid
loading heavy dependencies (pandas, numpy) during CLI --help rendering.
Dataset loaders are imported only when actually accessed.
"""

from typing import Any

__all__ = [
    "AdultIncomeDataset",
    "GermanCreditDataset",
    "get_adult_income_schema",
    "get_available_datasets",
    "get_german_credit_schema",
    "load_adult_income",
    "load_dataset",
    "load_german_credit",
    "validate_dataset_name",
]

# Available datasets metadata
AVAILABLE_DATASETS = {
    "german_credit": {
        "name": "German Credit",
        "description": "Credit risk assessment (1,000 samples, binary classification)",
        "loader": "load_german_credit",
    },
    "adult_income": {
        "name": "Adult Income",
        "description": "Income prediction (48,842 samples, binary classification)",
        "loader": "load_adult_income",
    },
}


def get_available_datasets() -> dict[str, dict[str, str]]:
    """Get metadata for all available datasets.

    Returns:
        Dictionary mapping dataset keys to metadata (name, description, loader)
    """
    return AVAILABLE_DATASETS.copy()


def validate_dataset_name(dataset: str) -> tuple[bool, str | None]:
    """Validate dataset name and provide helpful suggestions.

    Args:
        dataset: Dataset name to validate

    Returns:
        Tuple of (is_valid, suggestion_message)
        - is_valid: True if dataset exists
        - suggestion_message: Error message with suggestions if invalid, None if valid
    """
    if dataset in AVAILABLE_DATASETS:
        return True, None

    # Check for case-insensitive match
    dataset_lower = dataset.lower()
    for key in AVAILABLE_DATASETS:
        if key.lower() == dataset_lower:
            return False, f"Dataset '{dataset}' not found. Did you mean '{key}'? (names are case-sensitive)"

    # Check for partial matches
    matches = [key for key in AVAILABLE_DATASETS if dataset_lower in key.lower() or key.lower() in dataset_lower]
    if matches:
        return False, f"Dataset '{dataset}' not found. Did you mean: {', '.join(matches)}?"

    # No match - show all available
    available = "\n".join(f"  • {key}: {info['description']}" for key, info in AVAILABLE_DATASETS.items())
    return (
        False,
        f"Dataset '{dataset}' not found.\n\nAvailable datasets:\n{available}\n\nTip: Dataset names are case-sensitive",
    )


def load_dataset(dataset_name: str) -> Any:
    """Load a dataset by name.

    Args:
        dataset_name: Name of the dataset to load ('german_credit' or 'adult_income')

    Returns:
        Loaded dataset (typically a pandas DataFrame)

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name == "german_credit":
        # Import on demand to avoid circular imports and heavy dependencies
        from .german_credit import load_german_credit

        return load_german_credit()
    elif dataset_name == "adult_income":
        # Import on demand to avoid circular imports and heavy dependencies
        from .adult_income import load_adult_income

        return load_adult_income()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: german_credit, adult_income")


def __getattr__(name: str) -> Any:
    """Lazy import dataset loaders to avoid heavy imports during CLI --help.

    This function is called when an attribute is not found in the module,
    allowing us to defer expensive imports until they're actually needed.
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
