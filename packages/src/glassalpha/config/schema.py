"""Configuration schema definitions using Pydantic.

This module defines the structure and validation for GlassAlpha
configuration files, ensuring type safety and consistency.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CalibrationConfig(BaseModel):
    """Probability calibration configuration."""

    model_config = ConfigDict(extra="forbid")

    method: str | None = Field(None, description="Calibration method ('isotonic' or 'sigmoid')")
    cv: int = Field(5, description="Number of cross-validation folds")
    ensemble: bool = Field(True, description="Whether to use ensemble calibration")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str | None) -> str | None:
        """Validate calibration method."""
        if v is None:
            return v
        v = v.lower()
        if v not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Calibration method must be 'isotonic' or 'sigmoid', got: {v}")
        return v

    @field_validator("cv")
    @classmethod
    def validate_cv(cls, v: int) -> int:
        """Validate CV folds."""
        if v < 2:
            raise ValueError(f"CV folds must be >= 2, got: {v}")
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Model type (xgboost, lightgbm, logistic_regression, etc.)")
    path: Path | None = Field(None, description="Path to saved model file")
    params: dict[str, Any] | None = Field(default_factory=dict, description="Additional model parameters")
    calibration: CalibrationConfig | None = Field(None, description="Optional probability calibration")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure model type is lowercase."""
        return v.lower()


class DataConfig(BaseModel):
    """Data configuration.

    Clean policy:
    1. data.dataset is required for any real run
    2. Use data.dataset="custom" with data.path for external files
    3. data_schema alone is allowed for schema-only utilities/tests
    """

    model_config = ConfigDict(extra="forbid")

    # Dataset specification (required for runs; "custom" enables path)
    dataset: str | None = Field(None, description="Dataset key from registry or 'custom' for external files")

    # Path specification (only valid when dataset == "custom")
    path: str | None = Field(None, description="Path to data file (only when dataset='custom')")

    # Fetch policy for automatic dataset downloading
    fetch: str = Field(
        "if_missing",
        description="Fetch policy: 'never', 'if_missing', or 'always'",
        pattern="^(never|if_missing|always)$",
    )

    # Offline mode (disables network operations)
    offline: bool = Field(False, description="Disable network operations for offline environments")

    # Schema specification (for validation utilities and tests)
    schema_path: Path | None = Field(None, description="Path to data schema file")
    data_schema: dict[str, Any] | None = Field(None, description="Inline data schema")

    # Feature configuration
    protected_attributes: list[str] = Field(
        default_factory=list,
        description="List of protected/sensitive attributes for fairness analysis",
    )
    intersections: list[str] = Field(
        default_factory=list,
        description="Two-way intersectional analysis specifications (e.g., ['gender*race', 'age*income'])",
    )
    target_column: str | None = Field(None, description="Name of target column")
    feature_columns: list[str] | None = Field(None, description="List of feature columns to use")

    @field_validator("protected_attributes")
    @classmethod
    def lowercase_attributes(cls, v: list[str]) -> list[str]:
        """Ensure attribute names are lowercase."""
        return [attr.lower() for attr in v]

    @field_validator("intersections")
    @classmethod
    def validate_intersections(cls, v: list[str]) -> list[str]:
        """Validate and normalize intersection specifications.

        Each intersection must be in format 'attr1*attr2' for two-way analysis.
        """
        normalized = []
        for intersection in v:
            # Ensure lowercase
            intersection = intersection.lower()

            # Validate format
            parts = intersection.split("*")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid intersection format '{intersection}'. Must be 'attr1*attr2' for two-way intersections.",
                )

            # Validate non-empty parts
            if not all(part.strip() for part in parts):
                raise ValueError(
                    f"Invalid intersection '{intersection}'. Both attributes must be non-empty.",
                )

            # Normalize whitespace
            normalized.append("*".join(part.strip() for part in parts))

        return normalized

    @field_validator("fetch")
    @classmethod
    def validate_fetch_policy(cls, v: str) -> str:
        """Validate fetch policy values."""
        if v not in {"never", "if_missing", "always"}:
            raise ValueError(f"Fetch policy must be 'never', 'if_missing', or 'always', got: {v}")
        return v

    @field_validator("path")
    @classmethod
    def expand_user_path(cls, v: str | None) -> str | None:
        """Expand user home directory in paths."""
        if v is None:
            return None
        return str(Path(v).expanduser())

    @model_validator(mode="after")
    def enforce_dataset_policy(self) -> "DataConfig":
        """Enforce clean dataset policy.

        Rules:
        1. Schema-only configs are allowed (no dataset/path needed)
        2. For actual data use, dataset is required
        3. dataset="custom" requires path; other datasets forbid path
        4. offline=True is incompatible with fetch="always"
        """
        # Check offline + fetch="always" incompatibility first
        if self.offline and self.fetch == "always":
            raise ValueError("offline=True is incompatible with fetch='always'")

        # Schema-only configs are allowed (no dataset/path needed)
        if self.data_schema is not None and not (self.dataset or self.path):
            return self

        # For any actual data use, dataset is required
        if not self.dataset:
            raise ValueError(
                "data.dataset is required. Use data.dataset='custom' with data.path for external files.",
            )

        # Custom dataset requires path
        if self.dataset == "custom":
            if not self.path:
                raise ValueError("data.path is required when data.dataset='custom'.")
        # Registry datasets forbid path
        elif self.path:
            raise ValueError(
                f"data.path must be omitted for registry dataset '{self.dataset}'. "
                "Use data.dataset='custom' to provide a custom path.",
            )

        return self


class ExplainerConfig(BaseModel):
    """Explainer configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field("first_compatible", description="Selection strategy (first_compatible, best_score)")
    priority: list[str] = Field(default_factory=list, description="Ordered list of explainer preferences")
    config: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-explainer configuration")
    enabled: bool = Field(True, description="Whether to generate explanations")


class MetricCategory(BaseModel):
    """Configuration for a category of metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether to compute these metrics")
    metrics: list[str] = Field(default_factory=list, description="Metric names")
    config: dict[str, Any] = Field(default_factory=dict, description="Category-specific configuration")


class PerturbationConfig(BaseModel):
    """Configuration for adversarial perturbation sweeps (E6+)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to run perturbation sweeps")
    epsilon_values: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.1],
        description="Epsilon values for perturbation (default: 1%, 5%, 10%)",
    )
    threshold: float = Field(
        0.15,
        description="Threshold for gate logic (PASS if max_delta < threshold)",
    )
    seed: int = Field(42, description="Random seed for deterministic perturbations")

    @field_validator("epsilon_values")
    @classmethod
    def validate_epsilon_values(cls, v: list[float]) -> list[float]:
        """Validate epsilon values are positive."""
        if any(eps <= 0 for eps in v):
            raise ValueError("All epsilon values must be positive")
        return sorted(v)  # Sort for consistency

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is positive."""
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    model_config = ConfigDict(extra="forbid")

    performance: list[str] | MetricCategory = Field(
        default_factory=lambda: MetricCategory(metrics=["accuracy", "precision", "recall", "f1", "auc_roc"]),
        description="Performance metrics",
    )
    fairness: list[str] | MetricCategory = Field(
        default_factory=lambda: MetricCategory(metrics=["demographic_parity", "equal_opportunity"]),
        description="Fairness metrics",
    )
    drift: list[str] | MetricCategory = Field(
        default_factory=lambda: MetricCategory(metrics=["psi"]),
        description="Drift metrics",
    )
    stability: PerturbationConfig | None = Field(
        None,
        description="Stability testing configuration (perturbation sweeps)",
    )
    custom: dict[str, list[str]] | None = Field(None, description="Custom metric categories")

    @field_validator("performance", "fairness", "drift")
    @classmethod
    def convert_list_to_category(cls, v):
        """Convert list format to MetricCategory."""
        if isinstance(v, list):
            return MetricCategory(metrics=v)
        return v


class RecourseConfig(BaseModel):
    """Recourse configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to generate recourse")
    immutable_features: list[str] = Field(default_factory=list, description="Features that cannot be changed")
    monotonic_constraints: dict[str, str] = Field(
        default_factory=dict,
        description="Monotonic constraints (increase_only, decrease_only)",
    )
    cost_function: str = Field("weighted_l1", description="Cost function for optimization")
    max_iterations: int = Field(100, description="Maximum optimization iterations")

    @field_validator("monotonic_constraints")
    @classmethod
    def validate_constraints(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate constraint values."""
        valid_constraints = {"increase_only", "decrease_only", "fixed"}
        for feature, constraint in v.items():
            if constraint not in valid_constraints:
                raise ValueError(
                    f"Invalid constraint '{constraint}' for feature '{feature}'. Must be one of {valid_constraints}",
                )
        return v


class ThresholdConfig(BaseModel):
    """Threshold selection configuration."""

    model_config = ConfigDict(extra="forbid")

    policy: str = Field(default="youden", description="Threshold selection policy")
    threshold: float | None = Field(None, description="Fixed threshold value (for 'fixed' policy)")
    cost_fp: float | None = Field(None, description="False positive cost (for 'cost_sensitive' policy)")
    cost_fn: float | None = Field(None, description="False negative cost (for 'cost_sensitive' policy)")

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        """Validate threshold policy."""
        v = v.lower()
        valid_policies = {"youden", "fixed", "prevalence", "cost_sensitive"}
        if v not in valid_policies:
            raise ValueError(f"Invalid threshold policy: '{v}'. Must be one of: {valid_policies}")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        """Validate fixed threshold value."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got: {v}")
        return v

    @field_validator("cost_fp", "cost_fn")
    @classmethod
    def validate_costs(cls, v: float | None) -> float | None:
        """Validate cost values."""
        if v is not None and v < 0:
            raise ValueError(f"Cost must be non-negative, got: {v}")
        return v


class ReportConfig(BaseModel):
    """Report generation configuration."""

    model_config = ConfigDict(extra="ignore")  # Allow unknown fields like styling

    template: str = Field("standard_audit", description="Report template name")
    output_format: str = Field("pdf", description="Output format (pdf, html, json)")
    threshold: ThresholdConfig | None = Field(None, description="Threshold selection configuration")
    include_sections: list[str] = Field(
        default_factory=lambda: [
            "lineage",
            "data_schema",
            "global_explanations",
            "local_explanations",
            "fairness",
            "drift",
            "recourse",
            "assumptions",
        ],
        description="Report sections to include",
    )
    custom_branding: dict[str, Any] | None = Field(None, description="Custom branding configuration (enterprise only)")


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int | None = Field(42, description="Random seed for determinism")
    deterministic: bool = Field(True, description="Enforce deterministic behavior")
    capture_environment: bool = Field(True, description="Capture environment information")
    strict: bool = Field(False, description="Enable strict reproduction mode (may impact performance)")
    thread_control: bool = Field(False, description="Control thread counts for deterministic parallel processing")
    warn_on_failure: bool = Field(True, description="Warn if some determinism controls fail")

    @field_validator("random_seed")
    @classmethod
    def validate_seed(cls, v: int | None) -> int | None:
        """Validate seed is positive if provided."""
        if v is not None and v < 0:
            raise ValueError("Random seed must be non-negative")
        return v


class ManifestConfig(BaseModel):
    """Manifest generation configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether to generate manifest")
    include_git_sha: bool = Field(True, description="Include git commit SHA")
    include_config_hash: bool = Field(True, description="Include configuration hash")
    include_data_hash: bool = Field(True, description="Include data hash")
    include_model_hash: bool = Field(True, description="Include model hash")
    output_path: Path | None = Field(None, description="Path to save manifest (default: alongside report)")


class VersionPolicy(BaseModel):
    """Version compatibility policy for preprocessing artifacts."""

    model_config = ConfigDict(extra="forbid")

    require_exact_in_strict: bool = Field(
        True,
        description="Require exact version match (==) in strict mode",
    )
    allow_patch_non_strict: bool = Field(
        True,
        description="Allow patch version drift (e.g., 1.3.2 → 1.3.5) in non-strict mode",
    )
    allow_minor_in_strict: bool = Field(
        False,
        description="Allow minor version drift (e.g., 1.3.x → 1.5.x) in strict mode (risky)",
    )


class UnknownThresholds(BaseModel):
    """Thresholds for unknown category detection."""

    model_config = ConfigDict(extra="forbid")

    notice: float = Field(0.001, description="Notice threshold (>0.1% unknown categories)")
    warn: float = Field(0.01, description="Warning threshold (≥1% unknown categories)")
    fail: float = Field(0.05, description="Failure threshold (≥5% unknown categories in strict mode)")

    @field_validator("notice", "warn", "fail")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got: {v}")
        return v


class PreprocessingConfig(BaseModel):
    """Preprocessing artifact verification configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        "auto",
        description="Preprocessing mode: 'artifact' (use production preprocessing) or 'auto' (automatic preprocessing)",
    )
    artifact_path: Path | None = Field(None, description="Path to preprocessing artifact (joblib-serialized)")
    expected_file_hash: str | None = Field(
        None,
        description="Expected file hash (SHA256 or BLAKE2b) for integrity verification",
    )
    expected_params_hash: str | None = Field(
        None,
        description="Expected params hash (SHA256) for logical equivalence verification",
    )
    expected_sparse: bool | None = Field(
        None,
        description="Expected output sparsity (True for sparse matrices, False for dense)",
    )
    fail_on_mismatch: bool = Field(True, description="Fail if hash mismatch detected")
    version_policy: VersionPolicy = Field(
        default_factory=VersionPolicy,
        description="Version compatibility policy for sklearn/numpy/scipy",
    )
    thresholds: UnknownThresholds = Field(
        default_factory=UnknownThresholds,
        description="Thresholds for unknown category detection",
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate preprocessing mode."""
        v = v.lower()
        if v not in {"artifact", "auto"}:
            raise ValueError(f"Preprocessing mode must be 'artifact' or 'auto', got: {v}")
        return v

    @model_validator(mode="after")
    def validate_artifact_requirements(self) -> "PreprocessingConfig":
        """Validate artifact mode requires artifact_path."""
        if self.mode == "artifact" and self.artifact_path is None:
            raise ValueError("Preprocessing mode 'artifact' requires artifact_path to be specified")
        return self


class SecurityConfig(BaseModel):
    """Security configuration."""

    model_config = ConfigDict(extra="forbid")

    strict: bool = Field(False, description="Enable strict security mode")
    model_paths: dict[str, Any] = Field(
        default_factory=lambda: {
            "allowed_dirs": [".", "~/models", "./models"],
            "allow_remote": False,
            "require_hash": False,
            "max_size_mb": 256.0,
            "allow_symlinks": False,
            "allow_world_writable": False,
        },
        description="Model path security configuration",
    )
    yaml_loading: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_file_size_mb": 10.0,
            "max_depth": 20,
            "max_keys": 1000,
        },
        description="YAML loading security configuration",
    )
    logging: dict[str, Any] = Field(
        default_factory=lambda: {
            "sanitize_messages": True,
            "enable_json": False,
            "max_message_length": 10000,
        },
        description="Logging security configuration",
    )


class AuditConfig(BaseModel):
    """Main audit configuration."""

    model_config = ConfigDict(extra="forbid")

    # Required fields
    audit_profile: str = Field(..., description="Audit profile name (e.g., tabular_compliance)")
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataConfig = Field(..., description="Data configuration")

    def __init__(self, **data):
        # Profile defaults are applied in the loader, not here
        # Schema should only validate, not mutate inputs
        super().__init__(**data)

    # Optional fields with defaults
    explainers: ExplainerConfig = Field(default_factory=ExplainerConfig, description="Explainer configuration")
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Metrics configuration")
    recourse: RecourseConfig = Field(default_factory=RecourseConfig, description="Recourse configuration")
    report: ReportConfig = Field(default_factory=ReportConfig, description="Report configuration")
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig,
        description="Reproducibility configuration",
    )
    manifest: ManifestConfig = Field(default_factory=ManifestConfig, description="Manifest configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing artifact verification configuration",
    )

    # Mode flags
    strict_mode: bool = Field(False, description="Enable strict mode for regulatory compliance")

    # Additional metadata
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata for audit trail")

    @model_validator(mode="before")
    @classmethod
    def _compat_old_fields(cls, values: Any) -> Any:
        """Back-compatibility shim for old flat field names.

        Maps old schema format to new nested structure:
        - data_path -> data.path
        - model_config -> model
        - output -> report
        """
        if isinstance(values, dict):
            # Convert old data_path to new nested data structure
            if "data_path" in values and "data" not in values:
                values["data"] = {"path": values.pop("data_path")}

            # Convert old model_config to new model structure
            if "model_config" in values and "model" not in values:
                values["model"] = values.pop("model_config")

            # Convert old output to new report structure (drop path since it's now CLI-level)
            if "output" in values and "report" not in values:
                old_output = values.pop("output")
                if isinstance(old_output, dict):
                    # Extract report config fields, ignore path (now handled at CLI level)
                    report_config = {k: v for k, v in old_output.items() if k != "path"}
                    if report_config:
                        values["report"] = report_config
                    # Note: output.path is now passed directly to CLI, not in config
        return values

    @field_validator("audit_profile")
    @classmethod
    def validate_profile(cls, v: str) -> str:
        """Ensure profile name is lowercase."""
        return v.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditConfig":
        """Create from dictionary."""
        return cls(**data)
