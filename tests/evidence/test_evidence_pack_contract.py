"""Contract tests for evidence pack functionality.

Tests the public API boundaries and error conditions.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from glassalpha.evidence import create_evidence_pack, verify_evidence_pack


class TestEvidencePackExportContract:
    """Test evidence pack export contract."""

    def test_export_creates_all_required_artifacts(self, tmp_path):
        """Export creates all required artifacts in evidence pack."""
        # Create a mock audit report and manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_data = {
            "glassalpha_version": "0.3.0",
            "schema_version": "0.2.0",
            "performance": {"accuracy": 0.85},
            "fairness": {"demographic_parity_max_diff": 0.02},
        }
        manifest_path.write_text(json.dumps(manifest_data))

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Verify pack exists
        assert pack_path.exists()
        assert pack_path.suffix == ".zip"

        # Extract and verify contents
        with zipfile.ZipFile(pack_path, "r") as zipf:
            # Check required files are present
            required_files = [
                "audit_report.html",
                "audit_report.manifest.json",
                "canonical.jsonl",
                "SHA256SUMS.txt",
                "verification_instructions.txt",
                "policy_decision.json",
                "badge.svg",
            ]

            extracted_files = zipf.namelist()
            for required_file in required_files:
                assert required_file in extracted_files, f"Missing required file: {required_file}"

    def test_export_generates_deterministic_pack_name(self, tmp_path):
        """Pack name is deterministic based on manifest content."""
        # Create audit report and manifest
        report_path = tmp_path / "test_audit.html"
        report_path.write_text("<html>Test audit</html>")

        manifest_path = tmp_path / "test_audit.manifest.json"
        manifest_data = {
            "dataset_name": "test_dataset",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        manifest_path.write_text(json.dumps(manifest_data))

        # Export twice - should get same pack name
        pack_path1 = create_evidence_pack(report_path)
        pack_path2 = create_evidence_pack(report_path)

        # Both should have same stem (name without extension)
        assert pack_path1.stem == pack_path2.stem

    def test_export_fails_on_missing_report(self, tmp_path):
        """Export fails with clear error when report file doesn't exist."""
        missing_path = tmp_path / "missing_report.html"

        with pytest.raises(FileNotFoundError, match="Audit report not found"):
            create_evidence_pack(missing_path)

    def test_export_creates_badge_with_metrics(self, tmp_path):
        """Badge SVG includes performance and fairness metrics."""
        # Create audit report and manifest with metrics
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_data = {
            "performance": {"accuracy": 0.92},
            "fairness": {"demographic_parity_max_diff": 0.03},
        }
        manifest_path.write_text(json.dumps(manifest_data))

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Extract badge and verify content
        with zipfile.ZipFile(pack_path, "r") as zipf:
            badge_content = zipf.read("badge.svg").decode("utf-8")

        # Check badge contains metrics and status
        assert "PASS +" in badge_content or "FAIL X" in badge_content
        assert "Acc: 0.920" in badge_content
        assert "DP: 0.030" in badge_content

    def test_export_creates_canonical_jsonl(self, tmp_path):
        """canonical.jsonl contains correct artifact information."""
        # Create audit report and manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_data = {"test": "data"}
        manifest_path.write_text(json.dumps(manifest_data))

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Extract canonical.jsonl
        with zipfile.ZipFile(pack_path, "r") as zipf:
            canonical_content = zipf.read("canonical.jsonl").decode("utf-8")

        # Parse JSONL lines
        lines = [line.strip() for line in canonical_content.split("\n") if line.strip()]
        entries = [json.loads(line) for line in lines]

        # Verify structure
        for entry in entries:
            assert "path" in entry
            assert "sha256" in entry
            assert "size" in entry
            assert entry["sha256"].startswith("sha256:")

    def test_export_creates_valid_sha256sums(self, tmp_path):
        """SHA256SUMS.txt is in correct format for sha256sum -c."""
        # Create audit report
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Extract SHA256SUMS.txt
        with zipfile.ZipFile(pack_path, "r") as zipf:
            sha256sums_content = zipf.read("SHA256SUMS.txt").decode("utf-8")

        # Verify format (hash filename)
        lines = sha256sums_content.strip().split("\n")
        for line in lines:
            if line.strip():
                parts = line.split()
                assert len(parts) == 2, f"Invalid SHA256SUMS format: {line}"
                hash_part, filename_part = parts
                assert len(hash_part) == 64, f"Invalid hash length: {hash_part}"
                assert all(c in "0123456789abcdef" for c in hash_part), f"Invalid hex: {hash_part}"

    def test_export_creates_verification_instructions(self, tmp_path):
        """Verification instructions contain correct command and hash."""
        # Create audit report
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Extract verification instructions
        with zipfile.ZipFile(pack_path, "r") as zipf:
            instructions_content = zipf.read("verification_instructions.txt").decode("utf-8")

        # Verify instructions contain pack name and verification command
        assert pack_path.name in instructions_content
        assert "glassalpha verify-evidence-pack" in instructions_content
        assert "pip install glassalpha" in instructions_content

    def test_export_without_badge_skips_badge_generation(self, tmp_path):
        """Badge is not generated when include_badge=False."""
        # Create audit report and manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_path.write_text('{"test": "data"}')

        # Export without badge
        pack_path = create_evidence_pack(report_path, include_badge=False)

        # Extract and verify badge is missing
        with zipfile.ZipFile(pack_path, "r") as zipf:
            extracted_files = zipf.namelist()
            assert "badge.svg" not in extracted_files

    def test_export_with_custom_output_path(self, tmp_path):
        """Custom output path is respected."""
        # Create audit report
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        # Custom output path
        custom_path = tmp_path / "custom_pack.zip"

        # Export with custom path
        pack_path = create_evidence_pack(report_path, output_path=custom_path)

        # Verify custom path was used
        assert pack_path == custom_path
        assert pack_path.exists()


class TestEvidencePackVerificationContract:
    """Test evidence pack verification contract."""

    def test_verification_detects_missing_pack(self, tmp_path):
        """Verification fails when pack file doesn't exist."""
        missing_pack = tmp_path / "missing_pack.zip"

        result = verify_evidence_pack(missing_pack)

        assert not result.is_valid
        assert "Pack file not found" in result.errors[0]

    def test_verification_detects_missing_sha256sums(self, tmp_path):
        """Verification fails when SHA256SUMS.txt is missing."""
        # Create a pack without SHA256SUMS.txt
        pack_path = tmp_path / "incomplete_pack.zip"

        with zipfile.ZipFile(pack_path, "w") as zipf:
            zipf.writestr("test.txt", "test content")

        result = verify_evidence_pack(pack_path)

        assert not result.is_valid
        assert "SHA256SUMS.txt not found in pack" in result.errors[0]

    def test_verification_detects_missing_canonical_jsonl(self, tmp_path):
        """Verification fails when canonical.jsonl is missing."""
        # Create a pack without canonical.jsonl
        pack_path = tmp_path / "incomplete_pack.zip"

        with zipfile.ZipFile(pack_path, "w") as zipf:
            zipf.writestr("SHA256SUMS.txt", "abc123  test.txt\n")

        result = verify_evidence_pack(pack_path)

        assert not result.is_valid
        assert "canonical.jsonl not found in pack" in result.errors[0]

    def test_verification_detects_checksum_mismatch(self, tmp_path):
        """Verification detects when file checksums don't match."""
        # Create a pack with wrong checksum
        pack_path = tmp_path / "mismatched_pack.zip"

        with zipfile.ZipFile(pack_path, "w") as zipf:
            zipf.writestr("audit_report.html", "<html>Audit report</html>")
            zipf.writestr("audit_report.manifest.json", '{"test": "data"}')
            zipf.writestr("test.txt", "modified content")
            # Write wrong checksum
            zipf.writestr(
                "SHA256SUMS.txt", "wronghash  test.txt\nabc123  audit_report.html\ndef456  audit_report.manifest.json\n"
            )
            zipf.writestr(
                "canonical.jsonl",
                '{"path": "test.txt", "sha256": "wronghash"}\n{"path": "audit_report.html", "sha256": "abc123"}\n{"path": "audit_report.manifest.json", "sha256": "def456"}\n',
            )

        result = verify_evidence_pack(pack_path)

        assert not result.is_valid
        assert "Checksum mismatch for test.txt" in result.errors[0]

    def test_verification_detects_missing_audit_artifact(self, tmp_path):
        """Verification fails when no audit artifact is found."""
        # Create a pack without audit report
        pack_path = tmp_path / "no_audit_pack.zip"

        with zipfile.ZipFile(pack_path, "w") as zipf:
            zipf.writestr("SHA256SUMS.txt", "abc123  manifest.json\n")
            zipf.writestr("canonical.jsonl", '{"path": "manifest.json", "sha256": "abc123"}\n')
            zipf.writestr("manifest.json", '{"test": "data"}')

        result = verify_evidence_pack(pack_path)

        assert not result.is_valid
        assert "No audit report (.html or .pdf) found in pack" in result.errors[0]

    def test_verification_succeeds_for_valid_pack(self, tmp_path):
        """Verification succeeds for properly formed pack."""
        # Create a valid pack
        pack_path = tmp_path / "valid_pack.zip"

        # Create audit report and manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_data = {"performance": {"accuracy": 0.85}}
        manifest_path.write_text(json.dumps(manifest_data))

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Verify the pack we just created
        result = verify_evidence_pack(pack_path)

        assert result.is_valid
        assert result.artifacts_verified > 0
        assert len(result.errors) == 0
        assert result.sha256 is not None

    def test_verification_handles_corrupted_zip(self, tmp_path):
        """Verification handles corrupted ZIP files gracefully."""
        # Create corrupted ZIP file
        pack_path = tmp_path / "corrupted_pack.zip"
        pack_path.write_bytes(b"This is not a valid ZIP file")

        result = verify_evidence_pack(pack_path)

        assert not result.is_valid
        assert "corrupted or not a valid ZIP" in result.errors[0]


class TestDeterminismContract:
    """Test deterministic behavior of evidence pack operations."""

    def test_export_deterministic_with_source_date_epoch(self, tmp_path, monkeypatch):
        """Export is deterministic when SOURCE_DATE_EPOCH is set."""
        # Set SOURCE_DATE_EPOCH for deterministic timestamps
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1609459200")  # 2021-01-01 00:00:00 UTC

        # Create audit report and manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_data = {"dataset_name": "test"}
        manifest_path.write_text(json.dumps(manifest_data))

        # Export twice - should be identical
        pack_path1 = create_evidence_pack(report_path)
        pack_path2 = create_evidence_pack(report_path)

        # Read both packs and compare
        pack1_content = pack_path1.read_bytes()
        pack2_content = pack_path2.read_bytes()

        assert pack1_content == pack2_content

    def test_pack_hash_deterministic(self, tmp_path):
        """Pack SHA256 is deterministic for same input."""
        # Create audit report and manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        manifest_path = tmp_path / "audit_report.manifest.json"
        manifest_data = {"test": "data"}
        manifest_path.write_text(json.dumps(manifest_data))

        # Export twice
        pack_path1 = create_evidence_pack(report_path)
        pack_path2 = create_evidence_pack(report_path)

        # Compute hashes
        import hashlib

        hash1 = hashlib.sha256(pack_path1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(pack_path2.read_bytes()).hexdigest()

        assert hash1 == hash2
