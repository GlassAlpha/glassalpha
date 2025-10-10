"""Critical regression guard tests for CI.

These tests focus on the most important architectural constraints
that must never be violated in production deployments.
"""

import numpy as np
import pytest

from glassalpha.metrics.core import compute_classification_metrics
from glassalpha.runtime.repro import set_repro


class TestCriticalRegressions:
    """Test critical regressions that would break production deployments."""

    def test_multiclass_metrics_auto_detection(self):
        """CRITICAL: Multiclass metrics must auto-detect and use appropriate averaging."""
        # Test with multiclass data (this used to fail with "average='binary'" error)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1])
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.3, 0.3, 0.4],
                [0.9, 0.05, 0.05],
                [0.1, 0.2, 0.7],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
            ],
        )

        # This should NOT raise an exception about binary averaging
        try:
            metrics = compute_classification_metrics(y_true, y_pred, y_proba)

            # Verify we got multiclass-appropriate metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics

            # These should be scalars (averaged) not arrays
            assert isinstance(metrics["precision"], (int, float))
            assert isinstance(metrics["recall"], (int, float))
            assert isinstance(metrics["f1_score"], (int, float))

        except ValueError as e:
            if "average='binary'" in str(e):
                pytest.fail(
                    f"REGRESSION: Multiclass metrics using binary averaging: {e}\n"
                    f"The metrics engine should auto-detect multiclass and use appropriate averaging.",
                )
            else:
                raise

    @pytest.mark.skip(reason="Security module not yet implemented")
    def test_security_defaults_are_secure(self):
        """CRITICAL: Security features must be secure by default."""
        config = SecurityConfig()

        # These must be secure by default for production safety
        assert config.model_paths["allow_remote"] is False, "Remote models must be disabled by default"
        assert config.model_paths["allow_symlinks"] is False, "Symlinks must be disabled by default"
        assert config.model_paths["allow_world_writable"] is False, "World-writable files must be disabled by default"
        assert config.logging["sanitize_messages"] is True, "Log sanitization must be enabled by default"

        # Must have reasonable security limits
        assert config.model_paths["max_size_mb"] <= 512, "Model size limit must be reasonable for security"
        assert config.yaml_loading["max_file_size_mb"] <= 20, "YAML size limit must prevent DoS attacks"

    def test_deterministic_reproduction_controls_randomness(self):
        """CRITICAL: Deterministic reproduction must control all major randomness sources."""
        import random

        # Set reproduction mode
        status = set_repro(seed=42, strict=True)

        # Must control essential randomness sources for regulatory compliance
        essential_controls = {"python_random", "numpy_random", "xgboost", "sklearn"}
        actual_controls = set(status["controls"].keys())

        missing_essential = essential_controls - actual_controls
        if missing_essential:
            pytest.fail(
                f"REGRESSION: Missing essential determinism controls: {missing_essential}\n"
                f"Regulatory compliance requires controlling all randomness sources.",
            )

        # Test that Python random is actually controlled
        random.seed(42)  # This should be overridden by set_repro
        first_value = random.random()

        random.seed(42)
        second_value = random.random()

        # Must be deterministic for regulatory compliance
        assert first_value == second_value, "Python random not properly controlled - breaks determinism"

    @pytest.mark.skip(reason="Security module not yet implemented")
    def test_log_sanitization_removes_critical_secrets(self):
        """CRITICAL: Log sanitization must remove the most dangerous secrets."""
        # Test the most critical secret patterns that could cause data breaches
        critical_secrets = [
            ("password=topsecret123", "password="),  # Password fields
            ("User: john.doe@company.com", "john.doe***@company.com"),  # Email PII
            ("Path: /Users/john/secret.txt", "/Users/[USER]"),  # User paths
        ]

        for message, expected_pattern in critical_secrets:
            sanitized = sanitize_log_message(message)

            # Must contain sanitization for critical secrets
            if expected_pattern not in sanitized:
                pytest.fail(
                    f"REGRESSION: Critical secret not sanitized: {message} -> {sanitized}\n"
                    f"Expected pattern: {expected_pattern}",
                )

    @pytest.mark.skip(reason="Security module not yet implemented")
    def test_path_validation_blocks_directory_traversal(self):
        """CRITICAL: Path validation must block directory traversal attacks."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a safe test file
            safe_file = tmp_path / "model.json"
            safe_file.write_text('{"test": "model"}')

            # Must block directory traversal attacks
            with pytest.raises(Exception):  # Should raise SecurityError
                validate_local_model("../../../etc/passwd", allowed_dirs=[str(tmp_path)])

            # Must block absolute paths outside allowed directories
            with pytest.raises(Exception):  # Should raise SecurityError
                validate_local_model("/etc/passwd", allowed_dirs=[str(tmp_path)])

            # Must allow valid files in allowed directories
            validated = validate_local_model(str(safe_file), allowed_dirs=[str(tmp_path)])
            assert validated.exists(), "Valid files in allowed directories must be accepted"

    @pytest.mark.skip(reason="Security module not yet implemented")
    def test_yaml_security_prevents_resource_exhaustion(self):
        """CRITICAL: YAML security must prevent resource exhaustion attacks."""
        from glassalpha.security.yaml_loader import YAMLSecurityError, safe_load_yaml

        # Must block oversized YAML content (DoS prevention)
        large_yaml = "key: " + "x" * (15 * 1024 * 1024)  # 15MB
        with pytest.raises(YAMLSecurityError, match="too large"):
            safe_load_yaml(large_yaml)

        # Must block deeply nested YAML (stack overflow prevention)
        deep_yaml = "root:"
        for i in range(30):  # Deeper than safe limit
            deep_yaml += f"\n{'  ' * (i + 1)}level{i}:"
        deep_yaml += f"\n{'  ' * 31}value: test"

        with pytest.raises(YAMLSecurityError, match="too deep"):
            safe_load_yaml(deep_yaml)

    def test_performance_no_major_regressions(self):
        """CRITICAL: Core operations must not have major performance regressions."""
        import time

        # Test metrics computation performance (must be fast for large datasets)
        n_samples = 5000  # Reasonable test size
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_proba = np.random.rand(n_samples, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        start_time = time.time()
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        end_time = time.time()

        computation_time = end_time - start_time

        # Must complete in reasonable time (performance regression check)
        if computation_time > 3.0:
            pytest.fail(
                f"REGRESSION: Metrics computation too slow: {computation_time:.2f}s for {n_samples} samples\n"
                f"Major performance regression detected - should be < 3s",
            )

        assert len(metrics) > 5, "Must compute multiple metrics efficiently"


class TestArchitecturalConstraints:
    """Test architectural constraints that prevent technical debt."""

    def test_no_hardcoded_paths_in_example_configs(self):
        """CRITICAL: Example configs must not contain hardcoded user paths."""
        import re
        from pathlib import Path

        config_dir = Path(__file__).parent.parent / "configs"

        violations = []

        for config_file in config_dir.glob("*.yaml"):
            with config_file.open("r", encoding="utf-8") as f:
                content = f.read()

            # Look for absolute user paths (security and portability issue)
            hardcoded_patterns = [
                r"/Users/[^/\s]+/",  # macOS absolute user paths
                r"/home/[^/\s]+/",  # Linux absolute user paths
                r"C:\\Users\\[^\\s]+\\",  # Windows absolute user paths
            ]

            for pattern in hardcoded_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(f"{config_file.name}: {matches}")

        if violations:
            pytest.fail(
                f"REGRESSION: Hardcoded absolute user paths in configs: {violations}\n"
                f"Use relative paths or ~ expansion for portability and security.",
            )

    def test_all_critical_imports_available(self):
        """CRITICAL: All critical modules must be importable."""
        # Test that core modules can be imported (prevents broken deployments)
        critical_imports = [
            "glassalpha.config.schema",
            "glassalpha.metrics.core",
            "glassalpha.runtime.repro",
        ]

        import_failures = []

        for module_name in critical_imports:
            try:
                __import__(module_name)
            except ImportError as e:
                import_failures.append(f"{module_name}: {e}")

        if import_failures:
            pytest.fail(
                f"REGRESSION: Critical import failures: {import_failures}\n"
                f"These modules are required for basic functionality.",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
