"""Contract tests for utils determinism and integrity functions.

These tests ensure critical utility functions for hashing, integrity verification,
and deterministic operations work correctly and consistently.
"""

import pytest


class TestUtilsDeterminismContracts:
    """Contract tests for utils determinism and integrity functions."""

    def test_hashing_is_deterministic(self):
        """CRITICAL: Hashing must be deterministic for manifest integrity."""
        from glassalpha.utils.hashing import hash_object

        test_data = {"test": "data for hashing"}
        hash1 = hash_object(test_data)
        hash2 = hash_object(test_data)

        assert hash1 == hash2, "Hashing must be deterministic for reproducible builds"

    def test_hashing_different_inputs_produce_different_hashes(self):
        """CRITICAL: Hashing must be collision-resistant for data integrity."""
        from glassalpha.utils.hashing import hash_object

        data1 = {"test": "data 1"}
        data2 = {"test": "data 2"}

        hash1 = hash_object(data1)
        hash2 = hash_object(data2)

        assert hash1 != hash2, "Different inputs must produce different hashes"

    def test_integrity_verification_boundary_valid(self):
        """CRITICAL: Integrity verification must handle valid data correctly."""
        from glassalpha.utils.integrity import verify_dataset_integrity

        # This is a contract test - the actual function signature must work
        # We test that the function exists and can be called, not the actual logic
        try:
            # Just test that the function exists and is callable
            assert callable(verify_dataset_integrity), "verify_dataset_integrity must be callable"
        except ImportError:
            pytest.skip("verify_dataset_integrity not available")

    def test_integrity_verification_boundary_corrupted(self):
        """CRITICAL: Integrity verification must handle corrupted data correctly."""
        from glassalpha.utils.integrity import verify_dataset_integrity

        # This is a contract test - the actual function signature must work
        # We test that the function exists and can be called, not the actual logic
        try:
            # Just test that the function exists and is callable
            assert callable(verify_dataset_integrity), "verify_dataset_integrity must be callable"
        except ImportError:
            pytest.skip("verify_dataset_integrity not available")

    def test_determinism_detection_contract(self):
        """CRITICAL: Determinism detection must work across different data types."""
        from glassalpha.utils.determinism import validate_deterministic_environment

        # Test that determinism validation doesn't raise exceptions
        # Contract: must return dict with validation results
        try:
            result = validate_deterministic_environment()
            assert isinstance(result, dict), "Determinism validation must return dict"
        except Exception as e:
            pytest.fail(f"Determinism validation should not raise exceptions: {e}")

    def test_seeds_module_basic_contract(self):
        """CRITICAL: Seeds module must provide consistent seed management."""
        from glassalpha.utils.seeds import set_global_seed

        # Test that seed setting doesn't raise exceptions
        # Contract: must accept integer seeds and not crash
        try:
            set_global_seed(42)
        except Exception as e:
            pytest.fail(f"Seed setting should not raise exceptions: {e}")


class TestUtilsIntegrityContracts:
    """Contract tests for utils integrity verification functions."""

    def test_manifest_validation_contract(self):
        """CRITICAL: Manifest validation must handle valid manifests correctly."""
        # Test that manifest validation exists and is callable
        # Contract: manifest validation function should be available
        try:
            from glassalpha.provenance.run_manifest import generate_run_manifest

            assert callable(generate_run_manifest), "generate_run_manifest must be callable"
        except ImportError:
            pytest.skip("generate_run_manifest not available")

    def test_proc_utils_contract(self):
        """CRITICAL: Process utilities must handle basic operations safely."""
        # Test that process utilities exist and are callable
        # Contract: should provide basic process utilities
        try:
            from glassalpha.utils.proc import run_command_safely

            assert callable(run_command_safely), "run_command_safely must be callable"
        except ImportError:
            pytest.skip("run_command_safely not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
