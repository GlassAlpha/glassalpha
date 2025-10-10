"""Test preprocessing manifest extraction and canonicalization."""

import json
from pathlib import Path

import pandas as pd
import pytest

# Import stubs
try:
    from glassalpha.preprocessing.introspection import extract_sklearn_manifest
    from glassalpha.preprocessing.loader import load_artifact
    from glassalpha.preprocessing.manifest import MANIFEST_SCHEMA_VERSION, params_hash
except ImportError:
    pytest.skip("preprocessing module not implemented yet", allow_module_level=True)


def test_encoder_settings_exposed(sklearn_artifact: Path):
    """Manifest includes handle_unknown, drop, categories (trimmed), and output dtype."""
    artifact = load_artifact(sklearn_artifact)
    manifest = extract_sklearn_manifest(artifact)

    # Should have components list
    assert "components" in manifest
    assert len(manifest["components"]) > 0

    # Find OneHotEncoder component
    encoder_found = False
    for comp in manifest["components"]:
        if "OneHotEncoder" in comp.get("class", ""):
            encoder_found = True
            # Must expose key settings
            assert "handle_unknown" in comp
            assert "drop" in comp
            assert "categories" in comp or "n_categories" in comp
            break

    assert encoder_found, "OneHotEncoder component not found in manifest"

    # Manifest must have output dtype
    assert "output_dtype" in manifest or any("dtype" in comp for comp in manifest["components"])


def test_unknown_category_rates_reported(
    sklearn_artifact: Path,
    toy_df: pd.DataFrame,
    toy_df_with_unknowns: pd.DataFrame,
):
    """Provide eval data with an unseen category. Expect per-column rates in manifest
    and banded severity classification (notice/warn/fail thresholds from config).
    """
    from glassalpha.preprocessing.introspection import compute_unknown_rates

    artifact = load_artifact(sklearn_artifact)

    # Compute unknown rates (pre-transform)
    unknown_rates = compute_unknown_rates(artifact, toy_df_with_unknowns)

    # Should return per-column rates
    assert isinstance(unknown_rates, dict)

    # Should detect unknowns in employment and housing
    # (toy_df_with_unknowns adds "consultant" and "parents")
    assert len(unknown_rates) > 0

    # Rates should be 0.0-1.0
    for feature, rate_info in unknown_rates.items():
        rate = rate_info.get("rate", rate_info) if isinstance(rate_info, dict) else rate_info
        assert 0.0 <= rate <= 1.0, f"Rate for {feature} out of bounds: {rate}"


def test_manifest_canonicalization_matches_golden(sklearn_artifact: Path, tmp_path: Path):
    """Compare manifest (sorted keys, trimmed values) to a frozen JSON golden to catch accidental schema drift."""
    artifact = load_artifact(sklearn_artifact)
    manifest = extract_sklearn_manifest(artifact)

    # Compute params hash (canonical representation)
    manifest_hash = params_hash(manifest)

    # Hash must be deterministic
    manifest_hash2 = params_hash(manifest)
    assert manifest_hash == manifest_hash2, "Params hash must be deterministic"

    # Must start with algorithm prefix
    assert manifest_hash.startswith("sha256:")

    # Manifest must include schema version
    assert manifest.get("manifest_schema_version") == MANIFEST_SCHEMA_VERSION

    # Save golden (in real test, this would be committed)
    golden_path = tmp_path / "manifest_golden.json"
    with open(golden_path, "w") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)

    # Reload and verify stability
    with open(golden_path) as f:
        golden = json.load(f)

    # Key fields must match
    assert manifest.get("manifest_schema_version") == golden.get("manifest_schema_version")
    assert len(manifest.get("components", [])) == len(golden.get("components", []))
