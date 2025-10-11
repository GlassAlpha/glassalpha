"""Custom exceptions with machine-readable error codes.

Phase 5: User-friendly error messages with fix suggestions and docs links.

Error code format: GAE{category}{number}
- Category 1xxx: Input validation errors
- Category 2xxx: Data/result validation errors
- Category 4xxx: File operation errors

All errors include:
- code: Machine-readable error code (e.g., "GAE1001")
- message: Human-readable error description
- fix: Actionable fix suggestion (what to do)
- docs: URL to documentation (for details)
"""

from __future__ import annotations


class GlassAlphaError(Exception):
    """Base exception for all GlassAlpha errors.

    All GlassAlpha exceptions include:
    - code: Error code (e.g., "GAE1001")
    - message: Description of what went wrong
    - fix: How to fix it
    - docs: Link to documentation

    Examples:
        >>> try:
        ...     raise GlassAlphaError(
        ...         code="GAE1001",
        ...         message="Invalid protected_attributes format",
        ...         fix="Use dict with string keys mapping to arrays",
        ...         docs="https://glassalpha.com/errors/GAE1001"
        ...     )
        ... except GlassAlphaError as e:
        ...     print(e.code, e.message, e.fix)

    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        fix: str,
        docs: str | None = None,
    ) -> None:
        """Initialize error with code, message, fix, and optional docs link.

        Args:
            code: Error code (e.g., "GAE1001")
            message: Human-readable error description
            fix: Actionable fix suggestion
            docs: Optional URL to documentation

        """
        self.code = code
        self.message = message
        self.fix = fix
        self.docs = docs or f"https://glassalpha.com/errors/{code}"

        # Format error for display
        error_text = f"[{code}] {message}\n\nFix: {fix}\nDocs: {self.docs}"
        super().__init__(error_text)

    def __str__(self) -> str:
        """String representation with code, message, fix, and docs."""
        return f"[{self.code}] {self.message}\n\nFix: {self.fix}\nDocs: {self.docs}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"GlassAlphaError(code='{self.code}', message='{self.message}')"


# Input validation errors (1xxx)


class InvalidProtectedAttributesError(GlassAlphaError):
    """GAE1001: Invalid protected_attributes format."""

    def __init__(self, reason: str) -> None:
        """Initialize with specific reason for invalid format.

        Args:
            reason: Specific reason (e.g., "keys must be strings")

        """
        super().__init__(
            code="GAE1001",
            message=f"Invalid protected_attributes format: {reason}",
            fix="Use dict with string keys mapping to pandas Series or numpy arrays. Example: {'gender': gender_array, 'race': race_array}",
        )


class LengthMismatchError(GlassAlphaError):
    """GAE1003: Length mismatch between arrays."""

    def __init__(self, expected: int, got: int, name: str) -> None:
        """Initialize with expected and actual lengths.

        Args:
            expected: Expected length (n_samples)
            got: Actual length
            name: Name of mismatched array

        """
        super().__init__(
            code="GAE1003",
            message=f"Length mismatch: {name} has {got} samples, expected {expected}",
            fix=f"Ensure {name} has same length as X and y ({expected} samples)",
        )


class NonBinaryClassificationError(GlassAlphaError):
    """GAE1004: Non-binary classification not supported."""

    def __init__(self, n_classes: int) -> None:
        """Initialize with number of classes detected.

        Args:
            n_classes: Number of unique classes

        """
        super().__init__(
            code="GAE1004",
            message=f"Non-binary classification not supported (found {n_classes} classes)",
            fix="GlassAlpha currently supports binary classification only. For multi-class, use one-vs-rest approach or filter to two classes.",
        )


class UnsupportedMissingPolicyError(GlassAlphaError):
    """GAE1005: Unsupported missing_policy value."""

    def __init__(self, policy: str) -> None:
        """Initialize with unsupported policy.

        Args:
            policy: The unsupported policy name

        """
        super().__init__(
            code="GAE1005",
            message=f"Unsupported missing_policy: '{policy}'",
            fix="Use 'unknown' (map NaN to 'Unknown' category) or 'drop' (remove rows with NaN)",
        )


class NoPredictProbaError(GlassAlphaError):
    """GAE1008: Model has no predict_proba method."""

    def __init__(self, model_type: str) -> None:
        """Initialize with model type.

        Args:
            model_type: Type of model

        """
        super().__init__(
            code="GAE1008",
            message=f"Model {model_type} has no predict_proba() method",
            fix="Train model with probability estimates enabled. For sklearn: use probability=True. Or use from_predictions() with y_proba.",
        )


class AUCWithoutProbaError(GlassAlphaError):
    """GAE1009: Accessing AUC without probabilities."""

    def __init__(self, metric: str) -> None:
        """Initialize with metric name.

        Args:
            metric: Metric requiring probabilities (e.g., "roc_auc")

        """
        super().__init__(
            code="GAE1009",
            message=f"Metric '{metric}' requires y_proba (predicted probabilities)",
            fix="Either: (1) Use model with predict_proba(), or (2) Access metrics that don't require probabilities: accuracy, precision, recall, f1",
        )


class MultiIndexNotSupportedError(GlassAlphaError):
    """GAE1012: MultiIndex not supported."""

    def __init__(self, data_name: str) -> None:
        """Initialize with data name.

        Args:
            data_name: Name of data with MultiIndex (e.g., "X")

        """
        super().__init__(
            code="GAE1012",
            message=f"{data_name} has MultiIndex which is not supported",
            fix=f"Use {data_name}.reset_index(drop=True) to flatten the index before passing to GlassAlpha",
        )


# Data/result validation errors (2xxx)


class ResultIDMismatchError(GlassAlphaError):
    """GAE2002: Result ID doesn't match expected value."""

    def __init__(self, expected: str, got: str) -> None:
        """Initialize with expected and actual IDs.

        Args:
            expected: Expected result ID
            got: Actual result ID

        """
        super().__init__(
            code="GAE2002",
            message=f"Result ID mismatch: expected '{expected[:16]}...', got '{got[:16]}...'",
            fix="Result not reproducible. Check: (1) Random seed matches, (2) Data hashes match, (3) Model version matches, (4) Package versions match",
        )


class DataHashMismatchError(GlassAlphaError):
    """GAE2003: Data hash doesn't match expected value."""

    def __init__(self, data_name: str, expected: str, got: str) -> None:
        """Initialize with data name and hashes.

        Args:
            data_name: Name of data (e.g., "X", "y")
            expected: Expected hash
            got: Actual hash

        """
        super().__init__(
            code="GAE2003",
            message=f"Data hash mismatch for {data_name}: expected '{expected[:20]}...', got '{got[:20]}...'",
            fix=f"Data has changed since config was created. Verify {data_name} matches expected dataset.",
        )


# Data validation errors (2xxx)


class CategoricalDataError(GlassAlphaError):
    """Categorical data found in features without preprocessing.

    sklearn models require numeric features. Categorical columns must be
    encoded (one-hot, label, etc.) before model training and prediction.
    """

    def __init__(
        self,
        categorical_columns: list[str],
        *,
        fix: str | None = None,
        docs: str | None = None,
    ) -> None:
        """Initialize categorical data error.

        Args:
            categorical_columns: List of categorical column names
            fix: Override default fix message
            docs: Override default docs link

        """
        if fix is None:
            fix = (
                "Encode categorical columns before model training. "
                "Example: pd.get_dummies(df, columns=['category_col']). "
                "Or use sklearn's ColumnTransformer with OneHotEncoder."
            )

        if docs is None:
            docs = "https://glassalpha.com/guides/categorical-data"

        message = (
            f"Categorical columns found: {categorical_columns}. "
            "sklearn models require numeric features. "
            "Encode categorical data before training."
        )

        super().__init__(
            code="GAE2001",
            message=message,
            fix=fix,
            docs=docs,
        )


# File operation errors (4xxx)


class FileExistsError(GlassAlphaError):
    """GAE4001: File already exists and overwrite=False."""

    def __init__(self, path: str) -> None:
        """Initialize with file path.

        Args:
            path: Path to existing file

        """
        super().__init__(
            code="GAE4001",
            message=f"File already exists: {path}",
            fix="Use overwrite=True to replace existing file, or choose a different path",
        )


# Configuration errors (5xxx)


class ConfigError(GlassAlphaError):
    """GAE5001: Configuration validation error."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        """Initialize with validation error message.

        Args:
            message: Description of configuration error
            fix: Optional fix suggestion (auto-generated if not provided)

        """
        if fix is None:
            fix = "Check configuration file for missing or invalid fields. See docs for schema."

        super().__init__(
            code="GAE5001",
            message=message,
            fix=fix,
            docs="https://glassalpha.com/getting-started/configuration",
        )


# Component errors (3xxx)


class ModelError(GlassAlphaError):
    """GAE3001: Model loading or compatibility error."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        """Initialize with model error message.

        Args:
            message: Description of model error
            fix: Optional fix suggestion

        """
        if fix is None:
            fix = "Verify model type and path are correct. Check model was trained with supported library."

        super().__init__(
            code="GAE3001",
            message=message,
            fix=fix,
            docs="https://glassalpha.com/guides/models",
        )


class ExplainerError(GlassAlphaError):
    """GAE3002: Explainer unavailable or incompatible."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        """Initialize with explainer error message.

        Args:
            message: Description of explainer error
            fix: Optional fix suggestion

        """
        if fix is None:
            fix = "Check explainer compatibility with model type. Install required dependencies (shap, lime, etc.)."

        super().__init__(
            code="GAE3002",
            message=message,
            fix=fix,
            docs="https://glassalpha.com/guides/explainability",
        )


class MetricError(GlassAlphaError):
    """GAE3003: Metric computation error."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        """Initialize with metric error message.

        Args:
            message: Description of metric error
            fix: Optional fix suggestion

        """
        if fix is None:
            fix = "Check metric requirements (e.g., probabilities, binary classification, sufficient samples)."

        super().__init__(
            code="GAE3003",
            message=message,
            fix=fix,
            docs="https://glassalpha.com/reference/metrics",
        )


# Runtime errors (6xxx)


class DeterminismError(GlassAlphaError):
    """GAE6001: Determinism validation failed."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        """Initialize with determinism error message.

        Args:
            message: Description of determinism failure
            fix: Optional fix suggestion

        """
        if fix is None:
            fix = "Set explicit random seed, use deterministic=True, and pin dependency versions."

        super().__init__(
            code="GAE6001",
            message=message,
            fix=fix,
            docs="https://glassalpha.com/guides/determinism",
        )


class BootstrapError(GlassAlphaError):
    """GAE6002: Bootstrap computation error."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        """Initialize with bootstrap error message.

        Args:
            message: Description of bootstrap error
            fix: Optional fix suggestion

        """
        if fix is None:
            fix = "Reduce n_bootstrap or increase timeout. Check for infinite loops in metric computation."

        super().__init__(
            code="GAE6002",
            message=message,
            fix=fix,
            docs="https://glassalpha.com/guides/confidence-intervals",
        )


# Legacy exception aliases for backwards compatibility (remove in v1.0)


class ModelLoadError(ModelError):
    """Deprecated: Use ModelError instead."""

    def __init__(self, message: str = "Model loading failed") -> None:
        """Initialize with message.

        Args:
            message: Error message

        """
        super().__init__(message)


class ExplainerUnavailableError(ExplainerError):
    """Deprecated: Use ExplainerError instead."""

    def __init__(self, message: str = "Requested explainer is not available") -> None:
        """Initialize with message.

        Args:
            message: Error message

        """
        super().__init__(message)
