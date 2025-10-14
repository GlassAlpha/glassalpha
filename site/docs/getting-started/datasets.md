# Built-in datasets

GlassAlpha provides automatic dataset fetching and caching for common benchmark datasets used in ML compliance auditing. This feature eliminates manual data preparation and ensures reproducible audit results.

!!! info "Looking for more datasets?"
This page covers GlassAlpha's **built-in dataset system** (automatic fetching and caching).

    For using **external datasets**, see:

    - [Using Custom Data](custom-data.md) - Tutorial for using your own datasets

## Overview

The built-in dataset system automatically:

- **Downloads** datasets from public repositories when needed
- **Processes** raw data into ML-ready format
- **Caches** processed data for future use
- **Mirrors** data to requested locations
- **Handles** concurrent access safely

## Available datasets

Use `glassalpha datasets list` to see all available datasets:

```bash
glassalpha datasets list
# KEY              SCHEMA    DEFAULT_FILE
# german_credit    v1        german_credit_processed.csv
```

### German Credit dataset

The German Credit dataset is a canonical benchmark for credit risk assessment and fairness analysis.

- **Source**: UCI Machine Learning Repository
- **Records**: 1,000 loan applications
- **Features**: 20 demographic, financial, and loan characteristics
- **Target**: Binary credit risk classification (good/bad)
- **Protected Attributes**: Gender, age groups, foreign worker status

## Configuration

### Dataset specification

Configure datasets using the `data.dataset` field:

```yaml
data:
  dataset: german_credit
  fetch: if_missing
  offline: false
  target_column: credit_risk
  feature_columns:
    - checking_account_status
    - duration_months
    - credit_amount
    - savings_account
    - employment_duration
    - age_years
    - gender
  protected_attributes:
    - gender
```

### Fetch policies

Control when datasets are fetched:

- **`never`**: Never attempt to fetch (use existing files only)
- **`if_missing`** (default): Fetch only if file doesn't exist
- **`always`**: Always fetch (re-download even if file exists)

```yaml
data:
  dataset: german_credit
  fetch: if_missing # Default
```

### Offline mode

Disable network operations for air-gapped environments:

```yaml
data:
  dataset: german_credit
  offline: true # No network access
```

## Cache locations

Datasets are cached in OS-appropriate directories:

- **macOS**: `~/Library/Application Support/glassalpha/data/`
- **Linux**: `$XDG_DATA_HOME/glassalpha/data/` (or `~/.local/share/glassalpha/data/`)
- **Windows**: `%APPDATA%/glassalpha/data/`

Override with `GLASSALPHA_DATA_DIR` environment variable:

```bash
export GLASSALPHA_DATA_DIR="/custom/cache/location"
glassalpha datasets cache-dir  # Shows current location
```

## CLI commands

### List available datasets

```bash
glassalpha datasets list
# KEY              SCHEMA    DEFAULT_FILE
# german_credit    v1        german_credit_processed.csv
```

### Show dataset information

```bash
glassalpha datasets info german_credit
# Dataset: german_credit
# Schema version: v1
# Default file: german_credit_processed.csv
# Expected location: /Users/username/Library/Application Support/glassalpha/data/german_credit_processed.csv
# Currently exists: true
```

### Fetch dataset manually

```bash
glassalpha datasets fetch german_credit
# ✅ Dataset 'german_credit' fetched successfully
# 📁 Location: /Users/username/Library/Application Support/glassalpha/data/german_credit_processed.csv

# Force re-download
glassalpha datasets fetch german_credit --force

# Fetch to custom location
glassalpha datasets fetch german_credit --dest /tmp/custom_location.csv
```

## Path vs dataset configuration

### Recommended: dataset keys

```yaml
data:
  dataset: german_credit # Semantic reference
  fetch: if_missing
```

### Alternative: explicit paths

```yaml
data:
  path: "~/.glassalpha/data/german_credit_processed.csv"
  fetch: if_missing
```

The dataset key approach is preferred as it:

- Provides semantic meaning
- Enables automatic fetching
- Supports version management
- Works across different environments

## How it works

### Automatic resolution

When you specify a dataset:

1. **Resolve Path**: Convert dataset key to cache location
2. **Check Existence**: Verify if dataset is already cached
3. **Fetch if Needed**: Download and process if missing
4. **Mirror to Request**: Create hard link or copy to requested location

### Concurrent safety

Multiple processes can request the same dataset simultaneously:

- **File Locking**: Prevents race conditions during download
- **Atomic Operations**: Temporary files ensure no partial downloads
- **Cache Reuse**: Only one process downloads, others use cached result

### Cross-platform compatibility

The system handles filesystem differences:

- **Hard Links**: Used when possible for efficiency
- **Copy Fallback**: Used for cross-device filesystems
- **Directory Creation**: Automatically creates parent directories

## Environment variables

- `GLASSALPHA_DATA_DIR`: Override default cache location
- Standard OS cache directories used when not set

## Troubleshooting

### Common issues

**"Data file not found"**

- Ensure `data.dataset` or `data.path` is specified
- Check `glassalpha datasets info <dataset>` for cache status
- Verify network connectivity if using `offline: false`

**"Permission denied"**

- Check cache directory permissions
- Use `GLASSALPHA_DATA_DIR` to specify writable location

**"Offline mode enabled"**

- Set `offline: false` to enable network access
- Or provide files manually at specified paths

### Debug information

```bash
# Check cache location
glassalpha datasets cache-dir

# Verify dataset availability
glassalpha datasets info german_credit

# Test manual fetch
glassalpha datasets fetch german_credit --force
```

## Next steps

- **Use built-in datasets** - Perfect for quick testing and examples
- **Use [your own data](custom-data.md)** - Complete tutorial for custom datasets

## Adding new datasets

To add a new dataset to the built-in collection:

1. **Create Dataset Loader**: Implement download and processing logic in `src/glassalpha/datasets/loaders/`
2. **Add to Built-ins**: Add entry to the built-in dataset collection
3. **Update Documentation**: Add to this page and CLI help

See the source code for examples of dataset registration and implementation.

**Contributing datasets:** If you have a high-quality public dataset that would benefit the community, see our [contribution guide](../reference/contributing.md) or open an issue on GitHub.
