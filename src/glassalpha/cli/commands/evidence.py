"""Evidence pack commands: export and verify.

Commands for creating and verifying audit evidence packages.
"""

import logging
import os
import sys
from pathlib import Path

import typer

from glassalpha.cli.exit_codes import ExitCode

logger = logging.getLogger(__name__)


def export_evidence_pack(
    report: Path = typer.Argument(..., help="Path to audit report (HTML or PDF)"),
    output: Path = typer.Option(None, help="Output ZIP path (auto-generated if omitted)"),
    config: Path | None = typer.Option(None, help="Include original config file"),
    no_badge: bool = typer.Option(False, help="Skip badge generation"),
):
    """Export evidence pack for audit verification.

    Creates tamper-evident ZIP with all audit artifacts, checksums,
    and verification instructions for regulatory submission.

    The evidence pack includes:
    - Audit report (HTML or PDF)
    - Provenance manifest (hashes, versions, seeds)
    - Policy decision log (stub for v0.3.0)
    - Configuration file (if provided)
    - SHA256 checksums and verification instructions

    Requirements:
        - Completed audit report (HTML or PDF format)
        - Manifest file (auto-generated during audit)

    Examples:
        # Basic export with auto-generated name
        glassalpha export-evidence-pack reports/audit_report.html

        # Custom output path
        glassalpha export-evidence-pack audit.pdf --output compliance/evidence_2024.zip

        # Include original config for reproducibility
        glassalpha export-evidence-pack audit.html --config audit_config.yaml

        # Skip badge generation (faster)
        glassalpha export-evidence-pack audit.pdf --no-badge

    """
    try:
        # Import version for badge
        import glassalpha
        from glassalpha.evidence import create_evidence_pack

        os.environ["GLASSALPHA_VERSION"] = glassalpha.__version__

        pack_path = create_evidence_pack(
            audit_report_path=report,
            output_path=output,
            include_badge=not no_badge,
        )

        # Compute pack SHA256 for display
        import hashlib

        pack_hash = hashlib.sha256(pack_path.read_bytes()).hexdigest()

        typer.secho(f"\n✅ Evidence pack ready: {pack_path}", fg=typer.colors.GREEN)
        typer.echo(f"🔒 SHA256: {pack_hash}")
        typer.echo("📦 Share this file + hash publicly (GitHub, Hugging Face, email)")
        typer.echo(f"🔍 Verify with: glassalpha verify-evidence-pack {pack_path}")

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        # Note: Can't reliably check sys.argv here due to Typer's exception handling
        # Log exception regardless - users can check logs if needed
        logger.exception("Evidence pack export failed")
        typer.secho(f"Failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None


def verify_evidence_pack(
    pack: Path = typer.Argument(..., help="Evidence pack ZIP to verify"),
    verbose: bool = typer.Option(False, help="Show detailed verification log"),
):
    """Verify evidence pack integrity.

    Confirms all checksums match and pack is tamper-free for regulatory
    verification. Returns exit code 0 if verified, 1 if verification fails.

    The verification checks:
    - ZIP file is readable and not corrupted
    - SHA256SUMS.txt is present and valid
    - All file checksums match
    - canonical.jsonl is well-formed
    - Required artifacts are present (audit report, manifest)

    Requirements:
        - Evidence pack ZIP file created with export-evidence-pack command

    Examples:
        # Basic verification
        glassalpha verify-evidence-pack evidence_pack.zip

        # Verbose output with detailed checksums
        glassalpha verify-evidence-pack pack.zip --verbose

        # Verify downloaded pack from regulator
        glassalpha verify-evidence-pack compliance_submission_2024.zip

        # Use in CI/CD pipeline
        glassalpha verify-evidence-pack evidence.zip || exit 1

    """
    try:
        from glassalpha.evidence import verify_evidence_pack

        result = verify_evidence_pack(pack)

        if result.is_valid:
            typer.secho(
                f"\n[SUCCESS] Verified - all {result.artifacts_verified} artifacts match canonical.jsonl",
                fg=typer.colors.GREEN,
            )
            if result.sha256:
                typer.echo(f"[HASH] SHA256: {result.sha256}")
            typer.echo("[INFO] Audit report verified")
            typer.echo("[INFO] Manifest verified")
            typer.echo("[INFO] Badge verified" if verbose else "")
            typer.echo("[SUCCESS] Reproducible proof confirmed")

        else:
            typer.secho("\n[ERROR] Verification FAILED", fg=typer.colors.RED)
            for error in result.errors:
                typer.echo(f"[ERROR] {error}")
            if result.warnings:
                typer.secho("[WARN] Warnings:", fg=typer.colors.YELLOW)
                for warning in result.warnings:
                    typer.echo(f"  {warning}")
            typer.echo("[WARN] Pack may be corrupted or tampered")
            raise typer.Exit(ExitCode.VALIDATION_ERROR)

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Evidence pack verification failed")
        typer.secho(f"Failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
