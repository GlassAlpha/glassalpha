# Preprocessing Test Suite - TDD Spine

This test suite establishes the **contract-test spine** for preprocessing artifact verification. Tests were written **before implementation** to define acceptance criteria.

## Test Files

### `test_loader.py` - Artifact Loading & Hashing

- **test_hashes_present_and_stable**: Dual hash (file + params), params hash stable across repickling
- **test_missing_artifact_is_actionable_error**: Clear message with path
- **test_corrupted_artifact_is_actionable_error**: Clean error, not stack dump

### `test_validation.py` - Security & Contract Validation

- **test_rejects_unknown_transformer_class**: Class allowlist enforcement with FQCN in error
- **test_sparse_flag_detected_and_enforced**: Sparsity detection and mismatch error
- **test_feature_count_and_names_match_model**: Shape + names with minimal diff

### `test_manifest.py` - Manifest Extraction

- **test_encoder_settings_exposed**: OneHotEncoder params (drop, handle_unknown, categories)
- **test_unknown_category_rates_reported**: Pre-transform unknown detection with rates
- **test_manifest_canonicalization_matches_golden**: Schema stability protection

### `test_strict_mode.py` - Strict Mode Enforcement

- **test_strict_mode_fails_on_version_mismatch**: Version policy (exact/patch/minor)
- **test_strict_profile_blocks_auto_mode**: Hard block auto mode, fail fast
- **test_strict_mode_requires_hashes**: Require both file_hash and params_hash

## Current Status

Run tests with:

```bash
cd packages
source venv/bin/activate
python -m pytest tests/preprocessing/ -v
```

Expected: Most tests **FAIL** initially (red phase)
Goal: Make each test pass by implementing minimal functionality (green phase)

## TDD Flow

1. **Pick a failing test** (e.g., `test_hashes_present_and_stable`)
2. **Implement minimal code** to make it pass
3. **Run test** → should turn green
4. **Refactor** if needed
5. **Move to next test**

## Fixtures (conftest.py)

- `toy_df()`: 5 rows, 2 numeric + 2 categorical
- `sklearn_artifact()`: Fitted Pipeline (imputer + encoder + scaler)
- `corrupted_artifact()`: Junk bytes
- `mismatched_version_manifest()`: Fake version mismatch
- `toy_df_with_unknowns()`: Eval data with unseen categories
- `sparse_artifact()`: Explicit sparse_output=True
- `dense_artifact()`: Explicit sparse_output=False

## What NOT to TDD

These get tests **after** implementation:

- Report template rendering (cosmetic)
- CLI output formatting (cosmetic)
- Doctor check messages (nicety)
- Logging output (nicety)

TDD focuses on **boundaries** and **catastrophic failures**, not cosmetics.
