# DeterminismValidator

**Module**: `glassalpha.utils.determinism_validator`

Validate that audit outputs are deterministic across multiple runs.

---

## Overview

The `DeterminismValidator` runs audits multiple times and verifies that outputs are byte-identical. Critical for regulatory compliance and reproducible research.

**Use cases**:

- Pre-deployment validation
- CI/CD regression testing
- Troubleshooting non-determinism
- Cross-platform verification

---

## Quick Start

```python
from glassalpha.utils.determinism_validator import validate_audit_determinism
from pathlib import Path

# Validate audit determinism (3 runs)
report = validate_audit_determinism(
    config_path=Path("audit_config.yaml"),
    runs=3,
    seed=42,
    check_shap=True,
)

if report.is_deterministic:
    print(f"✅ Deterministic: {report.summary}")
else:
    print(f"❌ Non-deterministic: {report.summary}")
    for source in report.non_determinism_sources:
        print(f"  - {source}")
```

---

## API Reference

### DeterminismValidator

```python
class DeterminismValidator:
    """Validate that audit outputs are deterministic."""

    def __init__(
        self,
        glassalpha_cmd: str = "glassalpha",
        working_dir: Path | None = None,
    ):
        """Initialize validator.

        Args:
            glassalpha_cmd: Command to run glassalpha (default: "glassalpha")
            working_dir: Directory for temporary files (default: temp dir)
        """
```

#### validate_audit_determinism

```python
def validate_audit_determinism(
    self,
    config_path: Path,
    *,
    runs: int = 3,
    seed: int = 42,
    check_shap: bool = True,
) -> DeterminismReport:
    """Run audit multiple times and verify byte-identical outputs.

    Args:
        config_path: Path to audit configuration file
        runs: Number of audit runs to perform (default: 3)
        seed: Random seed for reproducibility (default: 42)
        check_shap: Verify SHAP available for deterministic explainer selection

    Returns:
        DeterminismReport with validation results

    Raises:
        DeterminismError: If prerequisites not met or validation fails

    Example:
        >>> validator = DeterminismValidator()
        >>> report = validator.validate_audit_determinism(
        ...     config_path=Path("config.yaml"),
        ...     runs=5,
        ...     seed=42,
        ... )
        >>> assert report.is_deterministic
    """
```

---

## DeterminismReport

```python
@dataclass
class DeterminismReport:
    """Results from determinism validation."""

    is_deterministic: bool
    """True if all runs produced identical outputs."""

    summary: str
    """Human-readable summary of results."""

    hashes: list[str]
    """Output hashes from each run."""

    run_times: list[float]
    """Execution time for each run (seconds)."""

    non_determinism_sources: list[str]
    """Identified sources of non-determinism (empty if deterministic)."""
```

**Example**:

```python
report = validator.validate_audit_determinism(config_path, runs=3)

print(f"Deterministic: {report.is_deterministic}")
print(f"Unique hashes: {len(set(report.hashes))}")
print(f"Average runtime: {sum(report.run_times) / len(report.run_times):.2f}s")

if not report.is_deterministic:
    print("Non-determinism sources:")
    for source in report.non_determinism_sources:
        print(f"  - {source}")
```

---

## Helper Functions

### validate_audit_determinism (convenience function)

```python
def validate_audit_determinism(
    config_path: Path,
    *,
    runs: int = 3,
    seed: int = 42,
    check_shap: bool = True,
) -> DeterminismReport:
    """Convenience function for quick validation.

    Equivalent to:
        validator = DeterminismValidator()
        return validator.validate_audit_determinism(...)

    Args:
        config_path: Path to audit configuration file
        runs: Number of audit runs (default: 3)
        seed: Random seed (default: 42)
        check_shap: Verify SHAP availability (default: True)

    Returns:
        DeterminismReport with validation results

    Example:
        >>> from glassalpha.utils.determinism_validator import validate_audit_determinism
        >>> report = validate_audit_determinism(Path("config.yaml"))
        >>> assert report.is_deterministic
    """
```

---

## Exceptions

### DeterminismError

```python
class DeterminismError(Exception):
    """Raised when determinism validation fails."""
```

**Common causes**:

- SHAP not installed (when `check_shap=True`)
- Missing environment variables
- Corrupt configuration file
- Insufficient disk space

---

## Usage Patterns

### CI/CD Integration

```python
import pytest
from glassalpha.utils.determinism_validator import validate_audit_determinism

def test_production_audit_is_deterministic():
    """Ensure production audit configuration is deterministic."""
    report = validate_audit_determinism(
        config_path=Path("production_config.yaml"),
        runs=5,  # More runs for higher confidence
        seed=42,
    )

    assert report.is_deterministic, (
        f"Production audit is non-deterministic:\n"
        f"{report.summary}\n"
        f"Sources: {report.non_determinism_sources}"
    )
```

### Pre-Deployment Checklist

```python
def validate_deployment_ready(config_path: Path) -> bool:
    """Validate audit is ready for production deployment."""

    # 1. Check determinism
    report = validate_audit_determinism(config_path, runs=3)
    if not report.is_deterministic:
        print(f"❌ Non-deterministic: {report.summary}")
        return False

    # 2. Check performance
    avg_time = sum(report.run_times) / len(report.run_times)
    if avg_time > 60:  # Must complete in <60s
        print(f"❌ Too slow: {avg_time:.1f}s (max 60s)")
        return False

    print(f"✅ Deployment ready:")
    print(f"  - Deterministic: {len(report.hashes)} identical hashes")
    print(f"  - Performance: {avg_time:.1f}s average")
    return True
```

### Troubleshooting Non-Determinism

```python
from glassalpha.utils.determinism_validator import DeterminismValidator

def debug_non_determinism(config_path: Path):
    """Debug sources of non-determinism."""
    validator = DeterminismValidator()

    # Run with detailed diagnostics
    report = validator.validate_audit_determinism(
        config_path=config_path,
        runs=10,  # More runs for pattern detection
        seed=42,
    )

    if report.is_deterministic:
        print("✅ Audit is deterministic")
        return

    print(f"❌ Non-deterministic audit detected")
    print(f"\nUnique hashes: {len(set(report.hashes))}")
    print(f"Hash distribution:")

    from collections import Counter
    hash_counts = Counter(report.hashes)
    for hash_val, count in hash_counts.most_common():
        print(f"  {hash_val[:16]}... : {count} times")

    print(f"\nIdentified sources:")
    for source in report.non_determinism_sources:
        print(f"  - {source}")

    print("\nRecommended fixes:")
    if "SHAP" in " ".join(report.non_determinism_sources):
        print("  1. Install SHAP: pip install shap")
    if "environment" in " ".join(report.non_determinism_sources).lower():
        print("  2. Set environment variables (see determinism guide)")
    if "seed" in " ".join(report.non_determinism_sources).lower():
        print("  3. Ensure explicit seed in config")
```

---

## Environment Requirements

The validator sets these environment variables automatically:

```python
{
    "SOURCE_DATE_EPOCH": "1577836800",  # Fixed timestamp
    "PYTHONHASHSEED": str(seed),        # Dict ordering
    "OMP_NUM_THREADS": "1",             # OpenMP
    "OPENBLAS_NUM_THREADS": "1",        # OpenBLAS
    "MKL_NUM_THREADS": "1",             # MKL
    "GLASSALPHA_DETERMINISTIC": "1",    # Strict mode
}
```

**Note**: These override any existing environment variables for validation runs.

---

## Performance

**Typical runtimes** (3 runs on German Credit dataset):

- Small model (<1000 samples): 15-30 seconds
- Medium model (1000-10000 samples): 30-90 seconds
- Large model (>10000 samples): 90-300 seconds

**Optimization tips**:

- Use fewer bootstrap samples in config (`n_bootstrap: 100` instead of `1000`)
- Reduce number of runs for quick checks (`runs=2`)
- Skip SHAP check if already verified (`check_shap=False`)

---

## Related Documentation

- [User Guide: Determinism](../../guides/determinism.md)
- [Troubleshooting Guide](../troubleshooting.md)

---

## Source

**Module path**: `src/glassalpha/utils/determinism_validator.py`

**Example notebook**: `examples/determinism_validation.ipynb`
