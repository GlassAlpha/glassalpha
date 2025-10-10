"""Determinism enforcement for byte-identical reproducibility.

This module provides comprehensive determinism enforcement to ensure
byte-identical outputs across runs and platforms. It addresses:

1. Random number generator seeding (Python, NumPy, ML frameworks)
2. BLAS/LAPACK threading (OpenBLAS, MKL, OpenMP)
3. PDF metadata normalization (timestamps, font order)
4. Platform-specific behavior (sorting, hashing, floating-point)

Usage:
    with deterministic(seed=42):
        # All operations are deterministic
        audit_results = run_audit(config)
        pdf = generate_report(audit_results)
"""

import hashlib
import logging
import os
import random
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from glassalpha.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)


def get_deterministic_timestamp(seed: int | None = None) -> datetime:
    """Get a deterministic timestamp for reproducible builds.

    Priority:
    1. SOURCE_DATE_EPOCH env var (CI/Docker builds)
    2. Seed-derived timestamp (local reproducibility)
    3. Current time (development only)

    Args:
        seed: Optional seed for deterministic timestamp generation

    Returns:
        datetime object (UTC) for use in manifests and metadata

    Example:
        >>> os.environ["SOURCE_DATE_EPOCH"] = "1577836800"  # 2020-01-01 00:00:00 UTC
        >>> get_deterministic_timestamp()
        datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

        >>> get_deterministic_timestamp(seed=42)
        datetime.datetime(2020, 1, 1, 0, 0, 42, tzinfo=datetime.timezone.utc)

    """
    # Priority 1: SOURCE_DATE_EPOCH (explicit override)
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        try:
            timestamp = int(source_date_epoch)
            return datetime.fromtimestamp(timestamp, tz=UTC)
        except (ValueError, OSError) as e:
            logger.warning(f"Invalid SOURCE_DATE_EPOCH '{source_date_epoch}': {e}")

    # Priority 2: Seed-based deterministic timestamp
    if seed is not None:
        # Use seed to generate deterministic timestamp
        # Base: 2020-01-01 00:00:00 UTC (epoch)
        base_timestamp = 1577836800  # 2020-01-01 00:00:00 UTC
        # Add seed modulo seconds in a year for variation
        offset = seed % (365 * 24 * 60 * 60)
        deterministic_timestamp = base_timestamp + offset
        return datetime.fromtimestamp(deterministic_timestamp, tz=UTC)

    # Priority 3: Current time (development mode)
    logger.debug("Using current time (non-deterministic). Set seed or SOURCE_DATE_EPOCH for reproducibility.")
    return datetime.now(UTC)


@contextmanager
def deterministic(seed: int, *, strict: bool = True) -> Generator[None, None, None]:
    """Context manager for deterministic execution.

    This enforces comprehensive determinism across:
    - Random number generators (Python, NumPy, optional frameworks)
    - BLAS threading (OpenBLAS, MKL, BLIS, OpenMP)
    - Hash randomization (PYTHONHASHSEED)
    - NumExpr parallelism

    Args:
        seed: Master seed for all randomness
        strict: If True, enforce single-threaded BLAS/LAPACK

    Yields:
        Context with deterministic environment

    Example:
        >>> with deterministic(seed=42):
        ...     results = run_audit(config)
        >>> # Results are byte-identical across runs

    Raises:
        RuntimeError: If critical environment variables cannot be set

    """
    # Save original environment
    original_env = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "BLIS_NUM_THREADS": os.environ.get("BLIS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "GLASSALPHA_DETERMINISTIC": os.environ.get("GLASSALPHA_DETERMINISTIC"),
    }

    # Save original random states
    original_random_state = random.getstate()
    original_numpy_state = np.random.get_state()

    try:
        # Set environment variables for determinism
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["GLASSALPHA_DETERMINISTIC"] = str(seed)  # Signal deterministic mode

        if strict:
            # Force single-threaded BLAS/LAPACK for determinism
            # These must be set before numpy/scipy imports, but we set them anyway
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["BLIS_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"

            logger.debug("Enforcing single-threaded BLAS/LAPACK for determinism")

        # Set all random seeds using centralized seed manager
        set_global_seed(seed)

        # Additional NumPy settings for determinism
        _enforce_numpy_determinism()

        logger.info(f"Deterministic context enabled with seed={seed}, strict={strict}")

        yield

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Restore random states
        random.setstate(original_random_state)
        np.random.set_state(original_numpy_state)

        logger.debug("Restored original random states and environment")


def _enforce_numpy_determinism() -> None:
    """Enforce NumPy-specific determinism settings."""
    # Set NumPy error handling to consistent mode
    np.seterr(all="warn")

    # Disable NumPy's implicit threading for consistent behavior
    # Note: This only affects numpy, not underlying BLAS
    try:
        # NumPy 2.0+ uses different API
        if hasattr(np, "set_num_threads"):
            np.set_num_threads(1)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not set NumPy threads: {e}")


def normalize_pdf_metadata(
    pdf_path: Path | str,
    *,
    fixed_timestamp: str = "2025-01-01T00:00:00Z",
    producer: str = "GlassAlpha/1.0",
    creator: str = "GlassAlpha",
) -> None:
    """Normalize PDF metadata for byte-identical reproducibility.

    This removes or fixes non-deterministic metadata:
    - CreationDate and ModDate (timestamps)
    - Producer string (may include version info)
    - Creator string
    - DocumentID (random UUID in some generators)

    Args:
        pdf_path: Path to PDF file to normalize
        fixed_timestamp: ISO8601 timestamp to use for dates
        producer: Fixed producer string
        creator: Fixed creator string

    Note:
        This modifies the PDF in-place. Make a backup if needed.
        Currently supports PDFs generated by WeasyPrint/Cairo.

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF format is not recognized

    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Read PDF content
    content = pdf_path.read_bytes()

    # Normalize metadata using pattern matching
    # This is a simple approach for WeasyPrint PDFs
    # For production, consider using pypdf or similar

    # Convert timestamp to PDF date format
    # PDF date format: D:YYYYMMDDHHmmSS
    from datetime import datetime

    dt = datetime.fromisoformat(fixed_timestamp.replace("Z", "+00:00"))
    pdf_date = dt.strftime("D:%Y%m%d%H%M%S")

    # Implement basic PDF metadata normalization
    # This handles the most common sources of non-determinism in PDFs

    try:
        # Try to use pypdf if available (optional dependency)
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, TextStringObject

        # Read the PDF
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()

        # Copy all pages
        for page in reader.pages:
            writer.add_page(page)

        # Normalize metadata
        if writer._info is None:
            writer._info = DictionaryObject()

        # Set fixed metadata values
        info = writer._info

        # Set fixed creation and modification dates
        info[NameObject("/CreationDate")] = TextStringObject(pdf_date)
        info[NameObject("/ModDate")] = TextStringObject(pdf_date)

        # Set fixed producer and creator
        info[NameObject("/Producer")] = TextStringObject(producer)
        info[NameObject("/Creator")] = TextStringObject(creator)

        # Remove or fix other potentially non-deterministic fields
        fields_to_remove = ["/ID", "/DocumentID", "/InstanceID", "/WebStatement"]
        for field in fields_to_remove:
            if field in info:
                del info[field]

        # Write back to file
        with open(pdf_path, "wb") as output_file:
            writer.write(output_file)

        logger.debug(f"Normalized PDF metadata for {pdf_path}")

    except ImportError:
        # Fallback: try basic text replacement (works for some PDF generators)
        try:
            content = pdf_path.read_bytes()

            # Replace common timestamp patterns
            import re

            # Pattern for PDF date format: D:YYYYMMDDHHMMSS
            date_pattern = rb"D:\d{14}"

            # Replace with fixed date
            fixed_content = re.sub(date_pattern, pdf_date.encode(), content)

            # Write back
            pdf_path.write_bytes(fixed_content)

            logger.debug(f"Applied basic PDF timestamp normalization for {pdf_path}")

        except Exception as e:
            logger.warning(f"Could not normalize PDF metadata: {e}. PDF may not be byte-identical.")

    except Exception as e:
        logger.warning(f"Error during PDF metadata normalization: {e}. PDF may not be byte-identical.")

    # Placeholder for future implementation
    # Will use pypdf to properly parse and rewrite metadata
    _ = pdf_date  # Silence unused variable warning
    _ = producer
    _ = creator
    _ = content


def compute_file_hash(file_path: Path | str, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of file for verification.

    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')

    Returns:
        Hexadecimal hash digest

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported

    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get hash function
    try:
        hash_func = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    # Read and hash file in chunks for memory efficiency
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def verify_deterministic_output(
    path1: Path | str,
    path2: Path | str,
    *,
    algorithm: str = "sha256",
) -> tuple[bool, str, str]:
    """Verify two files are byte-identical.

    Args:
        path1: First file path
        path2: Second file path
        algorithm: Hash algorithm to use

    Returns:
        Tuple of (are_identical, hash1, hash2)

    """
    hash1 = compute_file_hash(path1, algorithm=algorithm)
    hash2 = compute_file_hash(path2, algorithm=algorithm)

    return (hash1 == hash2, hash1, hash2)


def validate_deterministic_environment(*, strict: bool = False) -> dict[str, Any]:
    """Validate environment for deterministic execution.

    Args:
        strict: If True, fail on any non-deterministic settings

    Returns:
        Validation results with warnings and errors

    Raises:
        RuntimeError: If strict=True and critical issues found

    """
    results: dict[str, Any] = {
        "status": "pass",
        "warnings": [],
        "errors": [],
        "checks": {},
    }

    # Check PYTHONHASHSEED
    hashseed = os.environ.get("PYTHONHASHSEED")
    if hashseed is None or hashseed == "random":
        msg = "PYTHONHASHSEED not set or random - hash ordering non-deterministic"
        results["warnings"].append(msg)
        results["checks"]["pythonhashseed"] = False
    else:
        results["checks"]["pythonhashseed"] = True

    # Check BLAS threading
    threading_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
    for var in threading_vars:
        value = os.environ.get(var)
        if value is None:
            msg = f"{var} not set - BLAS may use multiple threads (non-deterministic)"
            results["warnings"].append(msg)
            results["checks"][var.lower()] = False
        elif value != "1":
            msg = f"{var}={value} - BLAS threading enabled (may be non-deterministic)"
            results["warnings"].append(msg)
            results["checks"][var.lower()] = False
        else:
            results["checks"][var.lower()] = True

    # Update status
    if results["errors"]:
        results["status"] = "fail"
    elif results["warnings"]:
        results["status"] = "warning"

    # Strict mode enforcement
    if strict and (results["errors"] or results["warnings"]):
        raise RuntimeError(
            f"Deterministic environment validation failed in strict mode:\n"
            f"Errors: {results['errors']}\n"
            f"Warnings: {results['warnings']}",
        )

    return results


def validate_audit_determinism(
    config_path: Path | str,
    output_path: Path | str,
    *,
    seed: int = 42,
    runs: int = 3,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate that audit generation is deterministic across multiple runs.

    This function runs the audit pipeline multiple times with the same seed
    and validates that all outputs are byte-identical.

    Args:
        config_path: Path to audit configuration file
        output_path: Path where audit report should be generated
        seed: Random seed for deterministic execution
        runs: Number of runs to perform for validation
        strict: If True, require byte-identical outputs

    Returns:
        Validation results with success status and details

    Raises:
        AssertionError: If outputs are not deterministic in strict mode
        RuntimeError: If audit generation fails

    """
    import subprocess

    config_path = Path(config_path)
    output_path = Path(output_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Set up deterministic environment
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "1577836800"  # Fixed timestamp for testing
    env["PYTHONHASHSEED"] = str(seed)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    results = {
        "success": True,
        "runs": runs,
        "hashes": [],
        "errors": [],
        "warnings": [],
    }

    # Run audit multiple times
    for run_num in range(runs):
        # Clean output file
        if output_path.exists():
            output_path.unlink()

        try:
            # Run audit command
            subprocess.run(
                ["glassalpha", "audit", "-c", str(config_path), "-o", str(output_path)],
                env=env,
                check=True,
                capture_output=True,
                encoding="utf-8",
            )

            # Verify output exists and compute hash
            if not output_path.exists():
                results["errors"].append(f"Output file not created in run {run_num + 1}")
                results["success"] = False
                continue

            if output_path.stat().st_size == 0:
                results["errors"].append(f"Output file empty in run {run_num + 1}")
                results["success"] = False
                continue

            # Compute hash
            file_hash = compute_file_hash(output_path)
            results["hashes"].append(file_hash)

        except subprocess.CalledProcessError as e:
            results["errors"].append(f"Audit failed in run {run_num + 1}: {e.stderr}")
            results["success"] = False
        except Exception as e:
            results["errors"].append(f"Unexpected error in run {run_num + 1}: {e}")
            results["success"] = False

    # Validate determinism
    if results["success"] and strict:
        unique_hashes = set(results["hashes"])
        if len(unique_hashes) != 1:
            results["warnings"].append(
                f"Outputs not deterministic: {len(unique_hashes)} unique hashes from {runs} runs: {results['hashes']}",
            )
            results["success"] = False

    return results
