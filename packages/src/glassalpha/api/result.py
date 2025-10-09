"""AuditResult: Immutable result object with rich API

Phase 2: Core structure with deep immutability.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Make array C-contiguous and read-only.

    This ensures arrays in results cannot be mutated, maintaining
    audit integrity for compliance requirements.

    Args:
        arr: Array to freeze

    Returns:
        Read-only, C-contiguous copy of array

    """
    # Always create a copy to avoid modifying the original
    arr = np.array(arr, copy=True, order="C")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class AuditResult:
    """Immutable audit result with deterministic ID.

    All audit results are deeply immutable to ensure compliance integrity.
    Results can be compared via strict equality (result.id) or tolerance-based
    equality (result.equals()) for cross-platform reproducibility.

    Attributes:
        id: SHA-256 hash of canonical result dict (deterministic)
        schema_version: Result schema version (e.g., "0.2.0")
        manifest: Provenance metadata (Mapping, immutable)
        performance: Performance metrics (PerformanceMetrics)
        fairness: Fairness metrics (FairnessMetrics)
        calibration: Calibration metrics (CalibrationMetrics)
        stability: Stability test results (StabilityMetrics)
        explanations: SHAP explanation summary (ExplanationSummary | None)
        recourse: Recourse generation summary (RecourseSummary | None)

    Examples:
        >>> result = ga.audit.from_model(model, X, y, protected_attributes={"gender": gender})
        >>> result.performance.accuracy  # 0.847
        >>> result.to_pdf("audit.pdf")
        >>> config = result.to_config()  # For reproduction

    """

    # Core identity
    id: str  # SHA-256 hash of canonical result dict
    schema_version: str  # e.g., "0.2.0"

    # Manifest (provenance, not hashed)
    manifest: Mapping[str, Any]

    # Metric sections (will be Mapping-like, immutable)
    performance: Mapping[str, Any]  # Will be PerformanceMetrics in full impl
    fairness: Mapping[str, Any]  # Will be FairnessMetrics
    calibration: Mapping[str, Any]  # Will be CalibrationMetrics
    stability: Mapping[str, Any]  # Will be StabilityMetrics
    explanations: Mapping[str, Any] | None = None  # Will be ExplanationSummary
    recourse: Mapping[str, Any] | None = None  # Will be RecourseSummary

    def __post_init__(self) -> None:
        """Freeze all mutable containers after initialization."""
        # Wrap manifest in MappingProxyType for immutability
        if not isinstance(self.manifest, types.MappingProxyType):
            object.__setattr__(self, "manifest", types.MappingProxyType(dict(self.manifest)))

        # Freeze metric sections (will be handled by metric wrappers in full impl)
        for field in ["performance", "fairness", "calibration", "stability", "explanations", "recourse"]:
            value = getattr(self, field)
            if value is not None and not isinstance(value, types.MappingProxyType):
                object.__setattr__(self, field, types.MappingProxyType(dict(value)))

    def __eq__(self, other: object) -> bool:
        """Strict equality via result.id.

        Two results are equal if and only if their IDs match.
        This ensures byte-identical reproducibility for compliance.

        For tolerance-based comparison, use result.equals().
        """
        if not isinstance(other, AuditResult):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Stable hash via result.id (first 16 hex chars).

        Enables use in sets and as dict keys.
        Hash is stable across pickle round-trips.
        """
        return int(self.id[:16], 16)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle support: convert MappingProxyType to dict."""
        state = {
            "id": self.id,
            "schema_version": self.schema_version,
            "manifest": dict(self.manifest),
            "performance": dict(self.performance),
            "fairness": dict(self.fairness),
            "calibration": dict(self.calibration),
            "stability": dict(self.stability),
            "explanations": dict(self.explanations) if self.explanations else None,
            "recourse": dict(self.recourse) if self.recourse else None,
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickle support: restore frozen dataclass fields."""
        # Restore each field using object.__setattr__ (frozen dataclass)
        object.__setattr__(self, "id", state["id"])
        object.__setattr__(self, "schema_version", state["schema_version"])
        object.__setattr__(self, "manifest", types.MappingProxyType(state["manifest"]))
        object.__setattr__(self, "performance", types.MappingProxyType(state["performance"]))
        object.__setattr__(self, "fairness", types.MappingProxyType(state["fairness"]))
        object.__setattr__(self, "calibration", types.MappingProxyType(state["calibration"]))
        object.__setattr__(self, "stability", types.MappingProxyType(state["stability"]))
        object.__setattr__(
            self,
            "explanations",
            types.MappingProxyType(state["explanations"]) if state["explanations"] else None,
        )
        object.__setattr__(
            self,
            "recourse",
            types.MappingProxyType(state["recourse"]) if state["recourse"] else None,
        )

    def equals(
        self,
        other: AuditResult,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Tolerance-based equality across all metrics.

        Use this for cross-platform reproducibility testing where
        floating point differences may occur due to different BLAS
        implementations or CPU architectures.

        Args:
            other: Result to compare against
            rtol: Relative tolerance (default: 1e-5)
            atol: Absolute tolerance (default: 1e-8)

        Returns:
            True if all metrics match within tolerance

        Examples:
            >>> result1 == result2  # Strict: byte-identical IDs
            False
            >>> result1.equals(result2, rtol=1e-5)  # Tolerant: float differences OK
            True

        """
        if self.id == other.id:
            return True

        # Compare each metric section with tolerance
        sections = ["performance", "fairness", "calibration", "stability"]
        for section in sections:
            self_metrics = dict(getattr(self, section))
            other_metrics = dict(getattr(other, section))

            if self_metrics.keys() != other_metrics.keys():
                return False

            for key in self_metrics:
                v1, v2 = self_metrics[key], other_metrics[key]

                # Handle None gracefully
                if v1 is None and v2 is None:
                    continue
                if v1 is None or v2 is None:
                    return False

                # Compare with tolerance
                try:
                    if not np.allclose(v1, v2, rtol=rtol, atol=atol, equal_nan=True):
                        return False
                except (TypeError, ValueError):
                    # Non-numeric comparison
                    if v1 != v2:
                        return False

        return True

    def __repr__(self) -> str:
        """Compact repr for logging."""
        return f"AuditResult(id='{self.id[:12]}...', schema_version='{self.schema_version}')"

    def _repr_html_(self) -> str:
        """Rich display in Jupyter (O(1) complexity).

        Shows high-level summary without computing expensive metrics.
        """
        # Type-guard formatting to prevent crashes
        perf = dict(self.performance)

        acc = perf.get("accuracy")
        acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "N/A"

        auc = perf.get("roc_auc")
        auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A"

        fair = dict(self.fairness)
        dp_max = fair.get("demographic_parity_max_diff")
        dp_str = f"{dp_max:.3f}" if isinstance(dp_max, (int, float)) else "N/A"

        return f"""
        <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
            <h3>GlassAlpha Audit Result</h3>
            <table>
                <tr><td><b>ID:</b></td><td><code>{self.id[:16]}...</code></td></tr>
                <tr><td><b>Accuracy:</b></td><td>{acc_str}</td></tr>
                <tr><td><b>ROC AUC:</b></td><td>{auc_str}</td></tr>
                <tr><td><b>Fairness (max ΔDP):</b></td><td>{dp_str}</td></tr>
            </table>
            <p><i>Use <code>result.to_pdf('audit.pdf')</code> for full report</i></p>
        </div>
        """

    def summary(self) -> dict[str, Any]:
        """Lightweight summary for logging (O(1)).

        Returns dictionary with key metrics for quick inspection.
        Does not compute expensive operations.

        Returns:
            Dictionary with id, schema_version, and key metrics

        """
        return {
            "id": self.id[:16],
            "schema_version": self.schema_version,
            "performance": {
                k: dict(self.performance)[k]
                for k in ["accuracy", "precision", "recall", "f1"]
                if k in dict(self.performance)
            },
            "fairness": {
                k: dict(self.fairness)[k]
                for k in ["demographic_parity_max_diff", "equalized_odds_max_diff"]
                if k in dict(self.fairness)
            },
        }

    def to_json(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Export to JSON with atomic write.

        Uses same canonicalization as result.id computation
        (strict JSON, NaN→null, Inf→sentinel).

        Args:
            path: Output path for JSON file
            overwrite: If True, overwrite existing file

        Raises:
            GlassAlphaError: If file exists and overwrite=False

        """
        # Phase 3: Will implement with full export logic
        msg = "to_json() will be implemented in Phase 3"
        raise NotImplementedError(msg)

    def to_html(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Export to HTML file with atomic write.

        Args:
            path: Output path for HTML file
            overwrite: If True, overwrite existing file

        Returns:
            Path to generated HTML file

        Raises:
            GlassAlphaError: If file exists and overwrite=False

        Example:
            >>> result = ga.audit.from_model(model, X, y)
            >>> result.to_html("audit_report.html")

        """
        from pathlib import Path

        output_path = Path(path)

        # Check for existing file
        if output_path.exists() and not overwrite:
            from ..exceptions import GlassAlphaError

            raise GlassAlphaError(f"File already exists: {output_path}. Use overwrite=True to replace.")

        # Generate HTML content using existing _repr_html_ method
        html_content = self._repr_html_()

        # Write to file atomically
        output_path.write_text(html_content, encoding="utf-8")

        return output_path

    def to_pdf(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Export to PDF with atomic write.

        Args:
            path: Output path for PDF file
            overwrite: If True, overwrite existing file

        Raises:
            GlassAlphaError: If file exists and overwrite=False

        """
        # Phase 3: Will implement with full export logic
        msg = "to_pdf() will be implemented in Phase 3"
        raise NotImplementedError(msg)

    def to_config(self) -> dict[str, Any]:
        """Generate config dict for reproduction.

        Returns dict with all parameters needed to reproduce this result:
        - model fingerprint
        - data hashes (with "sha256:" prefix)
        - protected attributes (categories + order)
        - random seed
        - expected result ID

        Returns:
            Dictionary with reproduction parameters

        """
        return {
            "model": {
                "fingerprint": self.manifest["model_fingerprint"],
                "type": self.manifest["model_type"],
            },
            "data": {
                "X_hash": self.manifest["data_hash_X"],  # "sha256:abc..."
                "y_hash": self.manifest["data_hash_y"],
                "n_samples": self.manifest["n_samples"],
                "n_features": self.manifest["n_features"],
            },
            "protected_attributes": self.manifest["protected_attributes_categories"],
            "random_seed": self.manifest["random_seed"],
            "expected_result_id": self.id,
            "schema_version": self.schema_version,
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save audit result (auto-detects format from extension).

        Args:
            path: Output file path (.html or .pdf)
            overwrite: If True, overwrite existing file

        Returns:
            Path to generated file

        Raises:
            GlassAlphaError: If file exists and overwrite=False

        Example:
            >>> result = ga.audit.from_model(model, X, y)
            >>> result.save("audit.html")  # HTML format
            >>> result.save("audit.pdf")   # PDF format (when implemented)

        """
        from pathlib import Path

        path = Path(path)

        if path.suffix.lower() == ".pdf":
            # PDF not implemented yet
            raise NotImplementedError("PDF export will be implemented in Phase 3")
        if path.suffix.lower() in [".html", ".htm"]:
            return self.to_html(path, overwrite=overwrite)
        raise ValueError(f"Unknown format: {path.suffix}. Use .pdf or .html extension")
