"""CLI commands for preprocessing artifact management.

This module provides commands for working with preprocessing artifacts,
including hashing, inspection, and validation.
"""

import json
import logging
from pathlib import Path

import typer

from .exit_codes import ExitCode

logger = logging.getLogger(__name__)

# Create Typer app for prep commands
prep_app = typer.Typer(
    name="prep",
    help="Preprocessing artifact management",
    no_args_is_help=True,
)


@prep_app.command("hash")
def compute_hash(
    artifact_path: Path = typer.Argument(
        ...,
        help="Path to preprocessing artifact (.joblib file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    show_params: bool = typer.Option(
        False,
        "--params",
        "-p",
        help="Also compute and show params hash",
    ),
):
    """Compute hash(es) for a preprocessing artifact.

    This command computes the file hash (SHA256) of a preprocessing artifact.
    Optionally, it can also compute the params hash (canonical hash of learned
    parameters) by loading and introspecting the artifact.

    Examples:
        # Just file hash (fast, no loading)
        glassalpha prep hash artifacts/preprocessor.joblib

        # File and params hash (slower, loads artifact)
        glassalpha prep hash artifacts/preprocessor.joblib --params

    """
    try:
        from glassalpha.preprocessing import (
            compute_file_hash,
            compute_params_hash,
            extract_sklearn_manifest,
            load_artifact,
        )

        typer.echo("Computing preprocessing artifact hashes...")
        typer.echo(f"Artifact: {artifact_path}")
        typer.echo()

        # Always compute file hash
        typer.echo("Computing file hash (SHA256)...")
        file_hash = compute_file_hash(artifact_path)
        typer.secho(f"✓ File hash: {file_hash}", fg=typer.colors.GREEN)

        # Optionally compute params hash
        if show_params:
            typer.echo()
            typer.echo("Loading artifact to compute params hash...")
            artifact = load_artifact(artifact_path)

            typer.echo("Extracting learned parameters...")
            manifest = extract_sklearn_manifest(artifact)

            typer.echo("Computing params hash (SHA256)...")
            params_hash = compute_params_hash(manifest)
            typer.secho(f"✓ Params hash: {params_hash}", fg=typer.colors.GREEN)

            typer.echo()
            typer.secho("Config snippet:", fg=typer.colors.BRIGHT_BLUE, bold=True)
            typer.echo("preprocessing:")
            typer.echo("  mode: artifact")
            typer.echo(f"  artifact_path: {artifact_path}")
            typer.echo(f"  expected_file_hash: '{file_hash}'")
            typer.echo(f"  expected_params_hash: '{params_hash}'")

    except ImportError as e:
        typer.secho(f"Error: Missing dependency - {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.USER_ERROR) from None
    except Exception as e:
        typer.secho(f"Error computing hashes: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.USER_ERROR) from None


@prep_app.command("inspect")
def inspect_artifact(
    artifact_path: Path = typer.Argument(
        ...,
        help="Path to preprocessing artifact (.joblib file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save manifest to JSON file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed component information",
    ),
):
    """Inspect a preprocessing artifact and display its manifest.

    This command loads the artifact, extracts all learned parameters,
    and displays a human-readable summary. Optionally saves the full
    manifest to a JSON file.

    Examples:
        # Quick inspection
        glassalpha prep inspect artifacts/preprocessor.joblib

        # Detailed inspection
        glassalpha prep inspect artifacts/preprocessor.joblib --verbose

        # Save manifest to file
        glassalpha prep inspect artifacts/preprocessor.joblib --output manifest.json

    """
    try:
        from glassalpha.preprocessing import extract_sklearn_manifest, load_artifact

        typer.echo(f"Loading artifact: {artifact_path}")
        artifact = load_artifact(artifact_path)

        typer.echo("Extracting manifest...")
        manifest = extract_sklearn_manifest(artifact)

        # Display summary
        typer.echo()
        typer.secho("=" * 60, fg=typer.colors.BRIGHT_BLUE)
        typer.secho("PREPROCESSING ARTIFACT MANIFEST", fg=typer.colors.BRIGHT_BLUE, bold=True)
        typer.secho("=" * 60, fg=typer.colors.BRIGHT_BLUE)
        typer.echo()

        # Schema version
        typer.echo(f"Manifest Schema Version: {manifest.get('manifest_schema_version', 'N/A')}")
        typer.echo()

        # Runtime versions
        artifact_versions = manifest.get("artifact_runtime_versions", {})
        if artifact_versions:
            typer.secho("Artifact Runtime Versions:", fg=typer.colors.CYAN, bold=True)
            for lib, version in artifact_versions.items():
                typer.echo(f"  {lib}: {version}")
            typer.echo()

        # Components
        components = manifest.get("components", [])
        typer.secho(f"Preprocessing Components ({len(components)}):", fg=typer.colors.CYAN, bold=True)
        typer.echo()

        for i, component in enumerate(components, 1):
            typer.secho(f"{i}. {component.get('name', 'Unknown')}", fg=typer.colors.YELLOW, bold=True)
            typer.echo(f"   Class: {component.get('class', 'N/A')}")

            # Show configuration
            if component.get("strategy"):
                typer.echo(f"   Strategy: {component.get('strategy')}")
            if component.get("handle_unknown"):
                typer.echo(f"   Handle Unknown: {component.get('handle_unknown')}")
            if "drop" in component:
                typer.echo(f"   Drop: {component.get('drop')}")
            if "sparse_output" in component:
                typer.echo(f"   Sparse Output: {component.get('sparse_output')}")

            # Show columns
            columns = component.get("columns", [])
            if columns:
                if verbose:
                    typer.echo(f"   Columns ({len(columns)}): {', '.join(columns)}")
                else:
                    typer.echo(f"   Columns: {len(columns)} columns")

            # Show learned parameters summary
            if verbose:
                if component.get("learned_stats"):
                    typer.echo(f"   Learned Stats: {len(component['learned_stats'])} values")
                if component.get("mean"):
                    typer.echo(f"   Scaling: mean + scale ({len(component['mean'])} features)")
                if component.get("categories"):
                    total_cats = sum(len(cats) for cats in component["categories"])
                    typer.echo(f"   Categories: {total_cats} total across {len(component['categories'])} columns")
                    if component.get("n_categories"):
                        typer.echo(f"   Category counts: {component['n_categories']}")

            typer.echo()

        # Save to file if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w") as f:
                json.dump(manifest, f, indent=2)
            typer.secho(f"✓ Manifest saved to: {output}", fg=typer.colors.GREEN)

    except ImportError as e:
        typer.secho(f"Error: Missing dependency - {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.USER_ERROR) from None
    except Exception as e:
        typer.secho(f"Error inspecting artifact: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.USER_ERROR) from None


@prep_app.command("validate")
def validate_artifact(
    artifact_path: Path = typer.Argument(
        ...,
        help="Path to preprocessing artifact (.joblib file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    expected_file_hash: str | None = typer.Option(
        None,
        "--file-hash",
        help="Expected file hash (sha256:...)",
    ),
    expected_params_hash: str | None = typer.Option(
        None,
        "--params-hash",
        help="Expected params hash (sha256:...)",
    ),
    check_versions: bool = typer.Option(
        True,
        "--check-versions/--no-check-versions",
        help="Check runtime version compatibility",
    ),
):
    """Validate a preprocessing artifact.

    This command performs comprehensive validation of a preprocessing artifact,
    including hash verification, class allowlisting, and version compatibility.

    Examples:
        # Basic validation (classes + versions)
        glassalpha prep validate artifacts/preprocessor.joblib

        # Validate with expected hashes
        glassalpha prep validate artifacts/preprocessor.joblib \\
            --file-hash sha256:abc123... \\
            --params-hash sha256:def456...

        # Skip version checks
        glassalpha prep validate artifacts/preprocessor.joblib --no-check-versions

    """
    try:
        from glassalpha.preprocessing import (
            assert_runtime_versions,
            compute_file_hash,
            compute_params_hash,
            extract_sklearn_manifest,
            load_artifact,
            validate_classes,
        )

        typer.echo(f"Validating artifact: {artifact_path}")
        typer.echo()

        # Step 1: File hash validation
        if expected_file_hash:
            typer.echo("1. Verifying file hash...")
            actual_file_hash = compute_file_hash(artifact_path)
            if actual_file_hash == expected_file_hash:
                typer.secho("   ✓ File hash matches", fg=typer.colors.GREEN)
            else:
                typer.secho("   ✗ File hash mismatch!", fg=typer.colors.RED, bold=True)
                typer.echo(f"   Expected: {expected_file_hash}")
                typer.echo(f"   Actual:   {actual_file_hash}")
                raise typer.Exit(code=ExitCode.USER_ERROR)
        else:
            typer.echo("1. File hash check skipped (no expected hash provided)")

        typer.echo()

        # Step 2: Load artifact
        typer.echo("2. Loading artifact...")
        artifact = load_artifact(artifact_path)
        typer.secho("   ✓ Artifact loaded successfully", fg=typer.colors.GREEN)
        typer.echo()

        # Step 3: Class allowlist validation
        typer.echo("3. Validating transformer classes...")
        try:
            validate_classes(artifact)
            typer.secho("   ✓ All classes are allowed", fg=typer.colors.GREEN)
        except ValueError as e:
            typer.secho(f"   ✗ Class validation failed: {e}", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=ExitCode.USER_ERROR)
        typer.echo()

        # Step 4: Params hash validation
        if expected_params_hash:
            typer.echo("4. Verifying params hash...")
            manifest = extract_sklearn_manifest(artifact)
            actual_params_hash = compute_params_hash(manifest)
            if actual_params_hash == expected_params_hash:
                typer.secho("   ✓ Params hash matches", fg=typer.colors.GREEN)
            else:
                typer.secho("   ✗ Params hash mismatch!", fg=typer.colors.RED, bold=True)
                typer.echo(f"   Expected: {expected_params_hash}")
                typer.echo(f"   Actual:   {actual_params_hash}")
                raise typer.Exit(code=ExitCode.USER_ERROR)
        else:
            typer.echo("4. Params hash check skipped (no expected hash provided)")

        typer.echo()

        # Step 5: Version compatibility
        if check_versions:
            typer.echo("5. Checking runtime version compatibility...")
            manifest = extract_sklearn_manifest(artifact)
            try:
                assert_runtime_versions(
                    manifest,
                    strict=False,
                    allow_minor=True,
                )
                typer.secho("   ✓ Runtime versions compatible", fg=typer.colors.GREEN)
            except (ValueError, RuntimeError) as e:
                typer.secho(f"   ⚠ Version warning: {e}", fg=typer.colors.YELLOW)
        else:
            typer.echo("5. Version check skipped")

        typer.echo()
        typer.secho("=" * 60, fg=typer.colors.GREEN)
        typer.secho("✓ VALIDATION PASSED", fg=typer.colors.GREEN, bold=True)
        typer.secho("=" * 60, fg=typer.colors.GREEN)

    except typer.Exit:
        raise
    except ImportError as e:
        typer.secho(f"Error: Missing dependency - {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.USER_ERROR) from None
    except Exception as e:
        typer.secho(f"Error validating artifact: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.USER_ERROR) from None
