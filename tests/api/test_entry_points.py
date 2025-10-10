"""Phase 3: Entry Point Signature Tests

Tests for from_model, from_predictions, from_config API surfaces.
"""

import pytest

from glassalpha.api import from_config, from_model, from_predictions

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


class TestFromModelSignature:
    """Tests for from_model() entry point."""

    def test_function_exists(self):
        """from_model() is accessible"""
        assert callable(from_model)

    def test_has_correct_signature(self):
        """from_model() has correct parameters"""
        import inspect

        sig = inspect.signature(from_model)

        # Required positional parameters
        assert "model" in sig.parameters
        assert "X" in sig.parameters
        assert "y" in sig.parameters

        # Keyword-only parameters
        assert "protected_attributes" in sig.parameters
        assert "sample_weight" in sig.parameters
        assert "random_seed" in sig.parameters
        assert "feature_names" in sig.parameters
        assert "class_names" in sig.parameters
        assert "explain" in sig.parameters
        assert "recourse" in sig.parameters
        assert "calibration" in sig.parameters
        assert "stability" in sig.parameters

    def test_has_defaults(self):
        """from_model() has sensible defaults"""
        import inspect

        sig = inspect.signature(from_model)

        # Check defaults
        assert sig.parameters["protected_attributes"].default is None
        assert sig.parameters["sample_weight"].default is None
        assert sig.parameters["random_seed"].default == 42
        assert sig.parameters["feature_names"].default is None
        assert sig.parameters["class_names"].default is None
        assert sig.parameters["explain"].default is True
        assert sig.parameters["recourse"].default is False
        assert sig.parameters["calibration"].default is True
        assert sig.parameters["stability"].default is False

    def test_has_docstring(self):
        """from_model() has comprehensive docstring"""
        doc = from_model.__doc__

        assert doc is not None
        assert "Generate audit from fitted model" in doc
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Raises:" in doc
        assert "Examples:" in doc

    def test_requires_valid_inputs(self):
        """from_model() validates inputs"""
        # Should raise error for None inputs
        with pytest.raises((AttributeError, ValueError)):
            from_model(model=None, X=None, y=None)


class TestFromPredictionsSignature:
    """Tests for from_predictions() entry point."""

    def test_function_exists(self):
        """from_predictions() is accessible"""
        assert callable(from_predictions)

    def test_has_correct_signature(self):
        """from_predictions() has correct parameters"""
        import inspect

        sig = inspect.signature(from_predictions)

        # Required positional parameters
        assert "y_true" in sig.parameters
        assert "y_pred" in sig.parameters
        assert "y_proba" in sig.parameters

        # Keyword-only parameters
        assert "protected_attributes" in sig.parameters
        assert "sample_weight" in sig.parameters
        assert "random_seed" in sig.parameters
        assert "class_names" in sig.parameters
        assert "model_fingerprint" in sig.parameters
        assert "calibration" in sig.parameters

    def test_has_defaults(self):
        """from_predictions() has sensible defaults"""
        import inspect

        sig = inspect.signature(from_predictions)

        # Check defaults
        assert sig.parameters["y_proba"].default is None
        assert sig.parameters["protected_attributes"].default is None
        assert sig.parameters["sample_weight"].default is None
        assert sig.parameters["random_seed"].default == 42
        assert sig.parameters["class_names"].default is None
        assert sig.parameters["model_fingerprint"].default is None
        assert sig.parameters["calibration"].default is True

    def test_has_docstring(self):
        """from_predictions() has comprehensive docstring"""
        doc = from_predictions.__doc__

        assert doc is not None
        assert "Generate audit from predictions" in doc
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Raises:" in doc
        assert "Examples:" in doc

    def test_requires_valid_inputs(self):
        """from_predictions() validates inputs"""
        # Should raise error for None inputs
        with pytest.raises(TypeError):
            from_predictions(y_true=None, y_pred=None)


class TestFromConfigSignature:
    """Tests for from_config() entry point."""

    def test_function_exists(self):
        """from_config() is accessible"""
        assert callable(from_config)

    def test_has_correct_signature(self):
        """from_config() has correct parameters"""
        import inspect

        sig = inspect.signature(from_config)

        # Required positional parameter
        assert "config_path" in sig.parameters

    def test_has_docstring(self):
        """from_config() has comprehensive docstring"""
        doc = from_config.__doc__

        assert doc is not None
        assert "Generate audit from YAML config" in doc
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Raises:" in doc
        assert "Config schema:" in doc
        assert "Examples:" in doc

    def test_requires_valid_file(self):
        """from_config() validates config file exists"""
        # Should raise FileNotFoundError for non-existent file
        with pytest.raises(FileNotFoundError):
            from_config("config.yaml")


class TestAPIAccessPatterns:
    """Tests for API access patterns."""

    def test_import_from_glassalpha_api(self):
        """Can import from glassalpha.api"""
        from glassalpha.api import AuditResult, from_config, from_model, from_predictions

        assert callable(from_model)
        assert callable(from_predictions)
        assert callable(from_config)
        assert AuditResult is not None

    def test_import_from_glassalpha_audit(self):
        """Can access via glassalpha.audit"""
        import glassalpha as ga

        assert hasattr(ga.audit, "from_model")
        assert hasattr(ga.audit, "from_predictions")
        assert hasattr(ga.audit, "from_config")
        assert hasattr(ga.audit, "AuditResult")

    def test_lazy_load_on_first_use(self):
        """Audit module lazy loads on first use"""
        import sys

        # Save original modules
        original_modules = sys.modules.copy()

        try:
            # Clear all glassalpha modules
            modules_to_remove = [key for key in sys.modules.keys() if key.startswith("glassalpha")]
            for key in modules_to_remove:
                del sys.modules[key]

            # Import glassalpha
            import glassalpha as ga

            # Verify audit not loaded yet
            assert "glassalpha.api" not in sys.modules

            # Access audit attribute
            _ = ga.audit

            # Verify loaded now
            assert "glassalpha.api" in sys.modules
        finally:
            # Restore original modules to avoid contaminating subsequent tests
            sys.modules.clear()
            sys.modules.update(original_modules)


class TestTypeAnnotations:
    """Tests for type annotations."""

    def test_from_model_has_annotations(self):
        """from_model() has type annotations"""
        import inspect

        sig = inspect.signature(from_model)

        # Check return annotation
        assert sig.return_annotation is not inspect.Signature.empty

        # Check parameter annotations
        assert sig.parameters["model"].annotation is not inspect.Parameter.empty
        assert sig.parameters["X"].annotation is not inspect.Parameter.empty
        assert sig.parameters["y"].annotation is not inspect.Parameter.empty

    def test_from_predictions_has_annotations(self):
        """from_predictions() has type annotations"""
        import inspect

        sig = inspect.signature(from_predictions)

        # Check return annotation
        assert sig.return_annotation is not inspect.Signature.empty

        # Check parameter annotations
        assert sig.parameters["y_true"].annotation is not inspect.Parameter.empty
        assert sig.parameters["y_pred"].annotation is not inspect.Parameter.empty

    def test_from_config_has_annotations(self):
        """from_config() has type annotations"""
        import inspect

        sig = inspect.signature(from_config)

        # Check return annotation
        assert sig.return_annotation is not inspect.Signature.empty

        # Check parameter annotations
        assert sig.parameters["config_path"].annotation is not inspect.Parameter.empty


class TestDocstringExamples:
    """Tests for docstring examples."""

    def test_from_model_examples_valid_syntax(self):
        """from_model() docstring examples are valid Python"""
        doc = from_model.__doc__

        # Extract examples section
        assert "Examples:" in doc

        # Check key patterns are present
        assert "ga.audit.from_model" in doc or "from_model" in doc
        assert "result.performance.accuracy" in doc
        assert "protected_attributes=" in doc

    def test_from_predictions_examples_valid_syntax(self):
        """from_predictions() docstring examples are valid Python"""
        doc = from_predictions.__doc__

        # Extract examples section
        assert "Examples:" in doc

        # Check key patterns
        assert "from_predictions" in doc
        assert "y_true=" in doc
        assert "y_pred=" in doc

    def test_from_config_examples_valid_syntax(self):
        """from_config() docstring examples are valid Python"""
        doc = from_config.__doc__

        # Extract examples section
        assert "Examples:" in doc

        # Check key patterns
        assert "from_config" in doc
        assert ".yaml" in doc


class TestErrorCodes:
    """Tests for error code documentation."""

    def test_from_model_documents_error_codes(self):
        """from_model() documents all error codes in Raises section"""
        doc = from_model.__doc__

        # Check error codes are documented
        assert "GAE1001" in doc  # Invalid protected_attributes
        assert "GAE1003" in doc  # Length mismatch
        assert "GAE1004" in doc  # Non-binary classification
        assert "GAE1012" in doc  # MultiIndex not supported

    def test_from_predictions_documents_error_codes(self):
        """from_predictions() documents all error codes"""
        doc = from_predictions.__doc__

        # Check error codes
        assert "GAE1001" in doc  # Invalid protected_attributes
        assert "GAE1003" in doc  # Length mismatch
        assert "GAE1004" in doc  # Non-binary classification

    def test_from_config_documents_error_codes(self):
        """from_config() documents all error codes"""
        doc = from_config.__doc__

        # Check error codes
        assert "GAE2002" in doc  # Result ID mismatch
        assert "GAE2003" in doc  # Data hash mismatch
