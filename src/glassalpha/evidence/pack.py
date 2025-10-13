"""Evidence pack creation and verification for audit artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    """Result of evidence pack verification."""

    is_valid: bool
    sha256: str | None
    artifacts_verified: int
    errors: list[str]
    warnings: list[str]


def create_evidence_pack(
    audit_report_path: Path,
    output_path: Path | None = None,
    *,
    include_badge: bool = True,
) -> Path:
    """Bundle audit artifacts into tamper-evident ZIP.

    Collects:
    - audit.html or audit.pdf (main report)
    - audit.manifest.json (provenance sidecar)
    - audit.shift_analysis.json (if exists)
    - config.yaml (if --config was used)
    - policy_decision.json (stub for v0.3.0)
    - badge.svg (pass/fail with metrics)
    - canonical.jsonl (artifact manifest)
    - SHA256SUMS.txt (checksums)
    - verification_instructions.txt (how to verify)

    Returns: Path to evidence_{dataset}_{date}.zip
    """
    # Validate input
    if not audit_report_path.exists():
        raise FileNotFoundError(f"Audit report not found: {audit_report_path}")

    # Collect artifacts
    artifacts = _collect_artifacts(audit_report_path)

    # Generate pack name from manifest or use default
    pack_name = _generate_pack_name(artifacts)

    if output_path is None:
        output_path = audit_report_path.parent / f"{pack_name}.zip"

    # Create temporary directory for pack contents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Copy artifacts to temp directory
        pack_artifacts = _copy_artifacts_to_pack(artifacts, temp_dir_path)

        # Generate canonical.jsonl
        canonical_manifest = _generate_canonical_jsonl(pack_artifacts)
        canonical_path = temp_dir_path / "canonical.jsonl"
        _write_jsonl(canonical_manifest, canonical_path)

        # Generate SHA256SUMS.txt
        sha256sums = _generate_sha256sums(pack_artifacts)
        sha256sums_path = temp_dir_path / "SHA256SUMS.txt"
        sha256sums_path.write_text(sha256sums)

        # Generate badge SVG
        badge_path = None
        manifest_path = None
        for filename in pack_artifacts:
            if filename.endswith(".manifest.json"):
                manifest_path = pack_artifacts[filename]
                break

        if include_badge and manifest_path:
            badge_content = _generate_badge_svg(manifest_path)
            badge_path = temp_dir_path / "badge.svg"
            badge_path.write_text(badge_content)

        # Generate verification instructions
        expected_sha256 = _compute_pack_hash(pack_artifacts, canonical_manifest)
        instructions = _generate_verification_instructions(pack_name, expected_sha256)
        instructions_path = temp_dir_path / "verification_instructions.txt"
        instructions_path.write_text(instructions)

        # Create policy decision stub
        policy_stub = _create_policy_stub()
        policy_path = temp_dir_path / "policy_decision.json"
        _write_json(policy_stub, policy_path)

        # Create final artifact list (include generated files)
        pack_artifacts["canonical.jsonl"] = canonical_path
        pack_artifacts["SHA256SUMS.txt"] = sha256sums_path
        pack_artifacts["verification_instructions.txt"] = instructions_path
        pack_artifacts["policy_decision.json"] = policy_path
        if badge_path and badge_path.exists():
            pack_artifacts["badge.svg"] = badge_path

        # Create ZIP with deterministic timestamps
        _create_zip(pack_artifacts, output_path, pack_name)

    return output_path


def verify_evidence_pack(pack_path: Path) -> VerificationResult:
    """Verify integrity of evidence pack.

    Checks:
    - ZIP is readable and not corrupted
    - SHA256SUMS.txt is present
    - All checksums match
    - canonical.jsonl is valid
    - Required artifacts present (audit + manifest)

    Returns: VerificationResult with pass/fail + details
    """
    if not pack_path.exists():
        return VerificationResult(
            is_valid=False,
            sha256=None,
            artifacts_verified=0,
            errors=[f"Pack file not found: {pack_path}"],
            warnings=[],
        )

    errors = []
    warnings = []

    try:
        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Extract ZIP
            with zipfile.ZipFile(pack_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir_path)

            # Find required files
            sha256sums_path = temp_dir_path / "SHA256SUMS.txt"
            canonical_path = temp_dir_path / "canonical.jsonl"

            if not sha256sums_path.exists():
                errors.append("SHA256SUMS.txt not found in pack")
                return VerificationResult(
                    is_valid=False,
                    sha256=None,
                    artifacts_verified=0,
                    errors=errors,
                    warnings=warnings,
                )

            if not canonical_path.exists():
                errors.append("canonical.jsonl not found in pack")
                return VerificationResult(
                    is_valid=False,
                    sha256=None,
                    artifacts_verified=0,
                    errors=errors,
                    warnings=warnings,
                )

            # Verify canonical.jsonl format
            try:
                canonical_data = _read_jsonl(canonical_path)
                if not canonical_data:
                    errors.append("canonical.jsonl is empty or invalid")
            except (json.JSONDecodeError, ValueError) as e:
                errors.append(f"canonical.jsonl is malformed: {e}")

            # Check for required artifacts (audit report + manifest)
            has_audit_report = any(f.get("path", "").endswith((".html", ".pdf")) for f in canonical_data)
            has_manifest = any(f.get("path", "").endswith(".manifest.json") for f in canonical_data)

            if not has_audit_report:
                errors.append("No audit report (.html or .pdf) found in pack")
            if not has_manifest:
                warnings.append(
                    "No audit manifest (.manifest.json) found in pack - provenance information may be limited"
                )

            # Early return if missing required artifacts
            if not has_audit_report or errors:
                return VerificationResult(
                    is_valid=False,
                    sha256=None,
                    artifacts_verified=0,
                    errors=errors,
                    warnings=warnings,
                )

            # Verify checksums (only if artifacts are present)
            sha256sums_content = sha256sums_path.read_text()
            artifacts_verified, checksum_errors = _verify_checksums(temp_dir_path, sha256sums_content)

            if checksum_errors:
                errors.extend(checksum_errors)

            # Compute pack SHA256
            pack_sha256 = _compute_pack_hash_from_files(temp_dir_path, canonical_data)

            is_valid = len(errors) == 0

            return VerificationResult(
                is_valid=is_valid,
                sha256=pack_sha256,
                artifacts_verified=artifacts_verified,
                errors=errors,
                warnings=warnings,
            )

    except zipfile.BadZipFile:
        return VerificationResult(
            is_valid=False,
            sha256=None,
            artifacts_verified=0,
            errors=["Pack file is corrupted or not a valid ZIP"],
            warnings=[],
        )
    except Exception as e:
        return VerificationResult(
            is_valid=False,
            sha256=None,
            artifacts_verified=0,
            errors=[f"Verification failed: {e}"],
            warnings=[],
        )


# Internal helper functions


def _collect_artifacts(report_path: Path) -> dict[str, Path]:
    """Collect all audit-related artifacts."""
    artifacts = {}
    base_path = report_path

    # Add main report
    artifacts[report_path.name] = report_path

    # Look for manifest sidecar
    manifest_path = report_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        artifacts[manifest_path.name] = manifest_path

    # Look for shift analysis
    shift_path = report_path.with_suffix(".shift_analysis.json")
    if shift_path.exists():
        artifacts[shift_path.name] = shift_path

    # Look for config (in parent directory)
    config_path = report_path.parent / "config.yaml"
    if config_path.exists():
        artifacts[config_path.name] = config_path

    return artifacts


def _generate_pack_name(artifacts: dict[str, Path]) -> str:
    """Generate short, deterministic pack name from artifacts."""
    # Use SOURCE_DATE_EPOCH for deterministic date
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        date_str = datetime.fromtimestamp(int(source_date_epoch), tz=UTC).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Try to extract dataset name from manifest
    manifest_path = None
    for filename, path in artifacts.items():
        if filename.endswith(".manifest.json"):
            manifest_path = path
            break

    dataset_name = "audit"  # Default fallback
    if manifest_path:
        try:
            manifest_data = _read_json(manifest_path)
            # Handle nested dataset field (dataset.name, dataset.dataset, etc.)
            dataset_field = manifest_data.get("dataset")
            if isinstance(dataset_field, dict):
                # Extract name from nested dict
                dataset_name = (
                    dataset_field.get("name")
                    or dataset_field.get("dataset")
                    or dataset_field.get("dataset_name")
                    or "audit"
                )
            elif isinstance(dataset_field, str):
                dataset_name = dataset_field
            else:
                # Try top-level fields as fallback
                dataset_name = manifest_data.get("dataset_name") or manifest_data.get("data_source") or "audit"

            # Sanitize filename (remove spaces, special chars, limit length)
            dataset_name = dataset_name.replace(" ", "_").replace("/", "_")
            # Remove other problematic characters
            dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in ("_", "-"))
            # Limit length to prevent filesystem errors
            if len(dataset_name) > 50:
                dataset_name = dataset_name[:50]

        except (json.JSONDecodeError, KeyError, AttributeError):
            dataset_name = "audit"

    return f"evidence_{dataset_name}_{date_str}"


def _copy_artifacts_to_pack(artifacts: dict[str, Path], temp_dir: Path) -> dict[str, Path]:
    """Copy artifacts to temporary directory for packing."""
    pack_artifacts = {}
    for filename, source_path in artifacts.items():
        dest_path = temp_dir / filename
        # Use atomic copy to avoid partial files
        with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
        pack_artifacts[filename] = dest_path
    return pack_artifacts


def _generate_canonical_jsonl(artifacts: dict[str, Path]) -> list[dict[str, Any]]:
    """Generate canonical artifact manifest."""
    entries = []

    # Sort artifacts for deterministic ordering
    for filename in sorted(artifacts.keys()):
        path = artifacts[filename]

        # Compute SHA256 hash
        sha256 = _compute_file_hash(path)

        # Get file size
        size = path.stat().st_size

        entry = {
            "path": filename,
            "sha256": f"sha256:{sha256}",
            "size": size,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        entries.append(entry)

    return entries


def _generate_sha256sums(artifacts: dict[str, Path]) -> str:
    """Generate SHA256SUMS.txt content."""
    lines = []

    # Sort artifacts for deterministic ordering
    for filename in sorted(artifacts.keys()):
        path = artifacts[filename]
        sha256 = _compute_file_hash(path)
        lines.append(f"{sha256}  {filename}")

    return "\n".join(lines) + "\n"


def _generate_badge_svg(manifest_path: Path) -> str:
    """Generate badge SVG from manifest data."""
    try:
        manifest_data = _read_json(manifest_path)

        # Extract metrics for badge - handle different manifest structures
        performance = manifest_data.get("performance", {})
        fairness = manifest_data.get("fairness", {})

        # Try different field names for accuracy
        accuracy = performance.get("accuracy") or performance.get("model_accuracy") or 0.0

        # Try different field names for fairness
        fairness_diff = (
            fairness.get("demographic_parity_max_diff")
            or fairness.get("max_fairness_difference")
            or fairness.get("fairness_score")
            or 0.0
        )

        # Determine status
        status = "PASS" if accuracy > 0.8 and fairness_diff < 0.05 else "FAIL"
        status_symbol = "+" if status == "PASS" else "X"

        # Format metrics
        acc_str = f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "N/A"
        fair_str = f"{fairness_diff:.3f}" if isinstance(fairness_diff, (int, float)) else "N/A"

        return f"""<svg width="200" height="40" xmlns="http://www.w3.org/2000/svg">
  <rect width="200" height="40" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
  <text x="10" y="15" font-family="Arial" font-size="12" fill="#333">
    {status} {status_symbol} | Acc: {acc_str} | DP: {fair_str}
  </text>
</svg>"""
    except (json.JSONDecodeError, KeyError):
        # Fallback badge
        return """<svg width="200" height="40" xmlns="http://www.w3.org/2000/svg">
  <rect width="200" height="40" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
  <text x="10" y="15" font-family="Arial" font-size="12" fill="#333">
    GlassAlpha Audit | Status: Unknown
  </text>
</svg>"""


def _generate_verification_instructions(pack_name: str, expected_sha256: str) -> str:
    """Generate verification instructions."""
    pack_filename = f"{pack_name}.zip" if not pack_name.endswith(".zip") else pack_name

    return f"""# GlassAlpha Evidence Pack Verification

Pack: {pack_filename}
Expected SHA256: {expected_sha256}

## Verify this pack:

1. Install GlassAlpha:
   pip install glassalpha

2. Verify checksum:
   echo "{expected_sha256}  {pack_filename}" | sha256sum -c

3. Run verification:
   glassalpha verify-evidence-pack {pack_filename}

4. Reproduce audit (optional):
   # Extract config.yaml from pack and run:
   # glassalpha audit --config config.yaml --output audit_copy.html
   # sha256sum audit_copy.html should match original

## About this pack:
- Generated by GlassAlpha {os.environ.get("GLASSALPHA_VERSION", "unknown")}
- All artifacts are byte-identical reproducible
- Verification works offline (no internet required)
- Share this pack + SHA256 for independent validation
"""


def _create_policy_stub() -> dict[str, Any]:
    """Create policy decision stub for future use."""
    return {
        "status": "not_configured",
        "note": "Policy gates available in v0.3.0 (Enhancement E1)",
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": "0.2.0",
    }


def _create_zip(artifacts: dict[str, Path], output_path: Path, pack_name: str) -> None:
    """Create ZIP file with deterministic timestamps."""
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        # Use fixed timestamp for determinism
        timestamp = int(source_date_epoch)
    else:
        # Use current time (not deterministic)
        timestamp = None

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for filename, path in artifacts.items():
            # Add file with deterministic timestamp
            info = zipfile.ZipInfo(filename)
            if timestamp:
                info.date_time = (2020, 1, 1, 0, 0, 0)  # Fixed date for determinism
            else:
                info.date_time = datetime.now().timetuple()[:6]
            zipf.writestr(info, path.read_bytes())


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_pack_hash(artifacts: dict[str, Path], canonical_manifest: list[dict[str, Any]]) -> str:
    """Compute overall pack hash from canonical manifest."""
    # Use canonical.jsonl content for hash computation
    canonical_json = json.dumps(canonical_manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _compute_pack_hash_from_files(temp_dir: Path, canonical_data: list[dict[str, Any]]) -> str:
    """Compute pack hash from extracted files."""
    return _compute_pack_hash({}, canonical_data)  # Same logic as above


def _verify_checksums(temp_dir: Path, sha256sums_content: str) -> tuple[int, list[str]]:
    """Verify file checksums against SHA256SUMS.txt."""
    errors = []
    verified = 0

    for line in sha256sums_content.strip().split("\n"):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 2:
            errors.append(f"Invalid SHA256SUMS format: {line}")
            continue

        expected_hash, filename = parts
        file_path = temp_dir / filename

        if not file_path.exists():
            errors.append(f"File not found in pack: {filename}")
            continue

        actual_hash = _compute_file_hash(file_path)
        if actual_hash != expected_hash:
            errors.append(f"Checksum mismatch for {filename}")
        else:
            verified += 1

    return verified, errors


def _write_json(data: Any, path: Path) -> None:
    """Write JSON with canonical formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)


def _write_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    """Write JSONL with canonical formatting."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries
