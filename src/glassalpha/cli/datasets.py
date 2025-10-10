"""CLI commands for dataset management.

This module provides CLI commands for listing, inspecting, and fetching
datasets from the GlassAlpha dataset registry.
"""

from pathlib import Path
from typing import Any

import typer

from ..datasets.registry import REGISTRY
from ..utils.cache_dirs import resolve_data_root
from .exit_codes import ExitCode

app = typer.Typer(help="Manage built-in datasets")


@app.command("list")
def list_datasets():
    """List all available datasets in the registry."""
    if not REGISTRY:
        typer.echo("No datasets registered.")
        return

    typer.echo("Available datasets:")
    typer.echo("KEY              SCHEMA    DEFAULT_FILE")
    typer.echo("-" * 45)

    for key, spec in REGISTRY.items():
        typer.echo(f"{key: <15} {spec.schema_version: <9} {spec.default_relpath}")


@app.command("info")
def dataset_info(
    dataset: str = typer.Argument(..., help="Dataset key to inspect"),
    show_columns: bool = typer.Option(False, "--show-columns", help="Show dataset columns with types"),
    show_sample: bool = typer.Option(False, "--show-sample", help="Show first 5 rows of data"),
    show_stats: bool = typer.Option(False, "--show-stats", help="Show basic descriptive statistics"),
    suggest_config: bool = typer.Option(False, "--suggest-config", help="Generate starter config based on schema"),
):
    """Show detailed information about a specific dataset."""
    spec = REGISTRY.get(dataset)
    if not spec:
        typer.echo(f"Dataset '{dataset}' not found in registry.")
        typer.echo(f"Available datasets: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(code=ExitCode.USER_ERROR)

    cache_root = resolve_data_root()
    expected_path = cache_root / spec.default_relpath

    # Log cache directory info for user visibility
    import os

    raw_env = os.getenv("GLASSALPHA_DATA_DIR")
    if raw_env:
        typer.echo(f"Cache dir: {raw_env} → {cache_root}")
    else:
        typer.echo(f"Cache dir: {cache_root}")

    typer.echo(f"Dataset: {spec.key}")
    typer.echo(f"Schema version: {spec.schema_version}")
    typer.echo(f"Default file: {spec.default_relpath}")
    typer.echo(f"Expected location: {expected_path}")
    typer.echo(f"Currently exists: {expected_path.exists()}")

    if spec.checksum:
        typer.echo(f"Checksum: {spec.checksum}")

    # Show columns if requested or if any other detailed flag is set
    if show_columns or show_sample or show_stats or suggest_config:
        if not expected_path.exists():
            typer.echo("\n❌ Dataset not downloaded yet. Use 'glassalpha datasets fetch' to download.")
            raise typer.Exit(code=ExitCode.USER_ERROR)

        try:
            import pandas as pd

            df = pd.read_csv(expected_path)

            # Get schema if available
            schema = _get_dataset_schema(dataset)

            if show_columns:
                _show_columns_info(df, schema, dataset)

            if show_sample:
                _show_sample_data(df)

            if show_stats:
                _show_statistics(df)

            if suggest_config:
                _suggest_config(df, schema, dataset)

        except Exception as e:
            typer.echo(f"\n❌ Failed to read dataset: {e}")
            raise typer.Exit(code=ExitCode.USER_ERROR)


@app.command("cache-dir")
def show_cache_dir():
    """Show the directory where datasets are cached."""
    import os

    cache_root = resolve_data_root()

    raw_env = os.getenv("GLASSALPHA_DATA_DIR")
    if raw_env:
        typer.echo(f"Cache dir: {raw_env} → {cache_root}")
    else:
        typer.echo(f"Cache dir: {cache_root}")


@app.command("fetch")
def fetch_dataset(
    dataset: str = typer.Argument(..., help="Dataset key to fetch"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if file exists"),
    dest: Path = typer.Option(None, "--dest", help="Custom destination path (overrides default cache location)"),
):
    """Fetch and cache a dataset from the registry."""
    spec = REGISTRY.get(dataset)
    if not spec:
        typer.echo(f"Dataset '{dataset}' not found in registry.")
        typer.echo(f"Available datasets: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(code=ExitCode.USER_ERROR)

    # Fetch the dataset directly using the spec
    from ..utils.cache_dirs import resolve_data_root

    cache_root = resolve_data_root()
    target_path = dest if dest else (cache_root / spec.default_relpath)
    target_path = Path(target_path).resolve()

    # Create parent directory if needed
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and force flag
    if target_path.exists() and not force:
        typer.echo(f"✅ Dataset '{dataset}' already cached")
        typer.echo(f"📁 Location: {target_path}")
        typer.echo("💡 Use --force to re-download")
        return

    try:
        # Fetch the dataset
        typer.echo(f"📥 Fetching dataset '{dataset}'...")

        # Call the fetch function (returns path where data was saved)
        fetched_path = spec.fetch_fn()

        # If dest was specified and different from where it was fetched, move it
        if dest and fetched_path != target_path:
            import shutil

            shutil.copy2(fetched_path, target_path)
            final_path = target_path
        else:
            final_path = fetched_path

        if not final_path.exists():
            raise FileNotFoundError("Fetch function did not create file at expected location")

        typer.echo(f"✅ Dataset '{dataset}' fetched successfully")
        typer.echo(f"📁 Location: {final_path}")

    except Exception as e:
        typer.echo(f"❌ Failed to fetch dataset '{dataset}': {e}")
        raise typer.Exit(code=ExitCode.USER_ERROR)


def _get_dataset_schema(dataset: str):
    """Get schema for a dataset if available."""
    try:
        if dataset == "german_credit":
            from ..datasets.german_credit import get_german_credit_schema

            return get_german_credit_schema()
        if dataset == "adult_income":
            from ..datasets.adult_income import get_adult_income_schema

            return get_adult_income_schema()
    except (ImportError, AttributeError):
        pass
    return None


def _show_columns_info(df, schema, dataset: str):  # noqa: ARG001
    """Show column information including types and categories."""
    typer.echo(f"\nColumns ({len(df.columns)}):")
    typer.echo("-" * 70)

    # Determine target and protected columns from schema
    target_col = None
    protected_cols = set()

    if schema:
        target_col = schema.target
        protected_cols = set(schema.sensitive_features) if hasattr(schema, "sensitive_features") else set()

    for col in df.columns:
        dtype = df[col].dtype
        dtype_str = str(dtype)

        # Shorten dtype names
        if dtype == object:
            dtype_str = "string"
        elif "int" in dtype_str:
            dtype_str = "integer"
        elif "float" in dtype_str:
            dtype_str = "float"

        # Add tags
        tags = []
        if col == target_col:
            tags.append("TARGET")
        if col in protected_cols:
            tags.append("PROTECTED")

        tag_str = f" [{', '.join(tags)}]" if tags else ""

        # Show unique values for categorical/object columns
        if dtype == object or df[col].nunique() < 20:
            unique_count = df[col].nunique()
            typer.echo(f"  {col: <30} {dtype_str: <10} ({unique_count} unique){tag_str}")
        else:
            typer.echo(f"  {col: <30} {dtype_str: <10}{tag_str}")

    # Show recommendations
    typer.echo("\n💡 Common configurations:")
    if target_col:
        typer.echo(f"  target_column: {target_col}")
    if protected_cols:
        typer.echo(f"  protected_attributes: {list(protected_cols)}")


def _show_sample_data(df):
    """Show first 5 rows of the dataset."""
    import pandas as pd  # noqa: PLC0415

    typer.echo("\nSample data (first 5 rows):")
    typer.echo("-" * 70)

    # Use pandas to_string for better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 20)

    typer.echo(df.head(5).to_string())


def _show_statistics(df):
    """Show basic descriptive statistics."""
    typer.echo("\nDescriptive statistics:")
    typer.echo("-" * 70)

    # Show row count and column types
    typer.echo(f"Rows: {len(df)}")
    typer.echo(f"Columns: {len(df.columns)}")

    # Count by dtype
    dtype_counts = df.dtypes.value_counts()
    typer.echo("\nColumn types:")
    for dtype, count in dtype_counts.items():
        typer.echo(f"  {dtype}: {count}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        typer.echo("\nMissing values:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            typer.echo(f"  {col}: {count} ({pct:.1f}%)")
    else:
        typer.echo("\nNo missing values ✓")

    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        typer.echo("\nNumeric columns (mean, std, min, max):")
        stats = df[numeric_cols].describe().loc[["mean", "std", "min", "max"]]
        typer.echo(stats.to_string())


def _suggest_config(df, schema, dataset: str):
    """Generate a suggested configuration based on dataset schema."""
    import yaml  # noqa: PLC0415

    typer.echo("\nSuggested configuration:")
    typer.echo("-" * 70)

    # Determine target and protected attributes
    target_col = None
    protected_attrs: list[str] = []

    if schema:
        target_col = schema.target if hasattr(schema, "target") else None
        if hasattr(schema, "sensitive_features"):
            protected_attrs = list(schema.sensitive_features)[:2]  # Take first 2

    # If no schema, try to infer
    if not target_col:
        # Look for common target column names
        target_candidates = ["target", "label", "y", "credit_risk", "income_over_50k", "outcome"]
        for col in df.columns:
            if col.lower() in target_candidates:
                target_col = col
                break

    if not protected_attrs:
        # Look for common protected attribute names
        protected_candidates = ["gender", "sex", "race", "age", "age_group", "ethnicity"]
        for col in df.columns:
            if any(cand in col.lower() for cand in protected_candidates):
                protected_attrs.append(col)
                if len(protected_attrs) >= 2:
                    break

    # Build config
    config: dict[str, Any] = {
        "audit_profile": "tabular_compliance",
        "data": {"dataset": dataset},
        "model": {"type": "xgboost", "params": {"random_state": 42, "n_estimators": 100}},
        "explainers": {
            "strategy": "first_compatible",
            "priority": ["treeshap", "kernelshap"],
        },
        "metrics": {
            "performance": ["accuracy", "f1_weighted"],
            "fairness": ["demographic_parity", "equal_opportunity"],
        },
        "reproducibility": {"random_seed": 42, "deterministic": True},
        "report": {"output_format": "html"},
        "manifest": {"enabled": True},
    }

    # Add target and protected attributes if found
    if target_col:
        config["data"]["target_column"] = target_col
    if protected_attrs:
        config["data"]["protected_attributes"] = protected_attrs

    # Print YAML
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    typer.echo(yaml_str)

    typer.echo("\n💡 Save this to a file:")
    typer.echo(f"  glassalpha datasets info {dataset} --suggest-config > audit_config.yaml")
