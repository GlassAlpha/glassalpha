"""Determinism validation framework for audit reproducibility.

This module provides tools to validate that audit outputs are byte-identical
across runs, which is critical for regulatory compliance and reproducible research.
"""

import hashlib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glassalpha.exceptions import DeterminismError

logger = logging.getLogger(__name__)


@dataclass
class DeterminismReport:
    """Report from determinism validation."""

    is_deterministic: bool
    hashes: list[str]
    summary: str
    non_determinism_sources: list[str]
    run_times: list[float]

    def __str__(self) -> str:
        """Human-readable summary of determinism validation."""
        status = "✅ DETERMINISTIC" if self.is_deterministic else "❌ NON-DETERMINISTIC"
        return f"{status}: {len(self.hashes)} runs, {self.summary}"


class DeterminismValidator:
    """Validator for audit output determinism."""

    def __init__(self) -> None:
        """Initialize determinism validator."""
        import shutil
        import sys

        # Use python3 -m glassalpha for consistency across environments
        # Store as list for subprocess.run()
        python3_path = shutil.which("python3")
        if python3_path:
            self.glassalpha_cmd = [python3_path, "-m", "glassalpha"]
        else:
            # Fallback to sys.executable if python3 not found
            self.glassalpha_cmd = [sys.executable, "-m", "glassalpha"]

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
            runs: Number of audit runs to perform
            seed: Random seed for deterministic execution
            check_shap: Whether to verify SHAP availability

        Returns:
            DeterminismReport with validation results

        Raises:
            DeterminismError: If prerequisites not met or validation fails

        """
        # Check prerequisites
        if check_shap:
            self._verify_shap_available()

        if not config_path.exists():
            raise DeterminismError(f"Config file not found: {config_path}")

        # Set up deterministic environment
        env = os.environ.copy()
        env.update(
            {
                "SOURCE_DATE_EPOCH": "1577836800",
                "PYTHONHASHSEED": str(seed),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "GLASSALPHA_DETERMINISTIC": "1",
                "TZ": "UTC",
                "MPLBACKEND": "Agg",
                # Additional determinism variables for WeasyPrint
                "WEASYPRINT_BACKEND": "cairo",
                "WEASYPRINT_DEBUG": "0",
            },
        )

        # Run audits and collect results
        hashes = []
        run_times = []
        non_determinism_sources = []

        for run_num in range(runs):
            logger.info("Running audit %d/%d for determinism validation", run_num + 1, runs)

            # Run single audit with same seed for reproducibility verification
            run_result = self._run_single_audit(config_path, seed, env)

            if run_result["success"]:
                hashes.append(run_result["hash"])
                run_times.append(run_result["duration"])
            else:
                non_determinism_sources.append(f"Run {run_num + 1}: {run_result['error']}")

        # Analyze results
        return self._analyze_determinism(hashes, run_times, non_determinism_sources)

    def _verify_shap_available(self) -> None:
        """Verify SHAP is available for deterministic explainer selection."""
        try:
            import shap  # noqa: F401
        except ImportError as e:
            raise DeterminismError(
                "SHAP required for deterministic explainer selection. Install with: pip install shap",
            ) from e

    def _run_single_audit(
        self,
        config_path: Path,
        seed: int,
        env: dict[str, str],
    ) -> dict[str, Any]:
        """Run a single audit and return hash and metadata.

        Args:
            config_path: Path to config file
            seed: Random seed for this run
            env: Environment variables

        Returns:
            Dict with success, hash, duration, and error info

        """
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create output path (use HTML for faster validation - PDF can be slow in CI)
            output_file = tmp_path / "audit.html"

            # Create a temporary config with the specific seed for this run
            run_config_path = tmp_path / "run_config.yaml"
            import yaml

            with config_path.open() as f:
                config_data = yaml.safe_load(f)

            # Add seed to reproducibility section
            if "reproducibility" not in config_data:
                config_data["reproducibility"] = {}
            config_data["reproducibility"]["random_seed"] = seed

            with run_config_path.open("w") as f:
                yaml.safe_dump(config_data, f)

            # Run audit
            start_time = time.time()

            try:
                # Build command list (glassalpha_cmd is already a list)
                cmd = self.glassalpha_cmd + [
                    "audit",
                    "-c",
                    str(run_config_path),
                    "-o",
                    str(output_file),
                    "--output-format",
                    "html",  # Use HTML for faster determinism validation
                ]

                subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout (increased for CI reliability)
                    check=True,
                )

                duration = time.time() - start_time

                # Compute hash of output
                if output_file.exists():
                    file_hash = self._compute_file_hash(output_file)
                    return {
                        "success": True,
                        "hash": file_hash,
                        "duration": duration,
                        "error": None,
                    }
                return {
                    "success": False,
                    "hash": None,
                    "duration": duration,
                    "error": "No output file generated",
                }

            except subprocess.CalledProcessError as e:
                duration = time.time() - start_time
                return {
                    "success": False,
                    "hash": None,
                    "duration": duration,
                    "error": f"CLI failed: {e.stderr}",
                }
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                return {
                    "success": False,
                    "hash": None,
                    "duration": duration,
                    "error": "Audit timed out after 10 minutes",
                }

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hash_obj = hashlib.sha256()

        with file_path.open("rb") as f:
            # Read in chunks for memory efficiency
            while chunk := f.read(8192):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _analyze_determinism(
        self,
        hashes: list[str],
        run_times: list[float],
        non_determinism_sources: list[str],
    ) -> DeterminismReport:
        """Analyze determinism validation results.

        Args:
            hashes: List of file hashes from successful runs
            run_times: Execution times for each run
            non_determinism_sources: Sources of non-determinism from failed runs

        Returns:
            DeterminismReport with analysis results

        """
        # Check if all successful runs have identical hashes
        if not hashes:
            return DeterminismReport(
                is_deterministic=False,
                hashes=hashes,
                summary="All runs failed - cannot validate determinism",
                non_determinism_sources=non_determinism_sources,
                run_times=run_times,
            )

        unique_hashes = set(hashes)
        is_deterministic = len(unique_hashes) == 1

        if is_deterministic:
            summary = f"All {len(hashes)} runs produced identical hash: {hashes[0][:12]}..."
        else:
            summary = f"Non-deterministic: {len(unique_hashes)} different hashes from {len(hashes)} runs"

        # Combine all non-determinism sources
        all_sources = non_determinism_sources.copy()

        return DeterminismReport(
            is_deterministic=is_deterministic,
            hashes=hashes,
            summary=summary,
            non_determinism_sources=all_sources,
            run_times=run_times,
        )


def validate_audit_determinism(
    config_path: Path,
    *,
    runs: int = 3,
    seed: int = 42,
    check_shap: bool = True,
) -> DeterminismReport:
    """Convenience function to validate audit determinism.

    Args:
        config_path: Path to audit configuration file
        runs: Number of runs to perform
        seed: Base seed for validation
        check_shap: Whether to require SHAP availability

    Returns:
        DeterminismReport with validation results

    """
    validator = DeterminismValidator()
    return validator.validate_audit_determinism(
        config_path=config_path,
        runs=runs,
        seed=seed,
        check_shap=check_shap,
    )
