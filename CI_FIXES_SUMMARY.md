# CI/CD Failure Fixes - October 12, 2025

## Summary

Three workflows failing with 5 distinct root causes. Priority fixes implemented for P0 blockers.

---

## ✅ P0 Fixes Implemented (Blocking All Workflows)

### 1. SBOM Generation Command Syntax

**Status:** ✅ FIXED
**File:** `.github/workflows/ci.yml:118`
**Change:**

```yaml
# Before (wrong)
cyclonedx-py --format json --outfile dist/sbom.json

# After (correct)
cyclonedx-py environment --format json --outfile dist/sbom.json
```

**Why:** `cyclonedx-py` requires subcommand (`environment`, `requirements`, etc.) before format options.

### 2. Template Rendering Error - Manifest Access

**Status:** ✅ FIXED
**Files:**

- `src/glassalpha/report/templates/standard_audit.html:716-717`
- `src/glassalpha/report/templates/standard_audit.html:2828-2829`
- `src/glassalpha/report/templates/standard_audit.html:2837`

**Root cause:** Template expects `manifest` as object with `.execution` attribute, but receives flat dict

**Changes:**

```jinja2
{# Before (wrong - object access) #}
manifest.execution.status
manifest.execution.duration_seconds

{# After (correct - dict access) #}
manifest.get('execution', {}).get('status')
manifest.get('execution', {}).get('duration_seconds')
```

**Why:** `manifest` is passed as dict in `src/glassalpha/report/context.py:202`. Template must use dict `.get()` syntax for safe access.

### 3. Test Fixture Schema Mismatch

**Status:** ✅ FIXED
**File:** `tests/integration/test_pdf_generation_linux.py:61-65`

**Root cause:** Test fixture uses `calibration_analysis` parameter removed from `AuditResults` schema

**Change:** Removed `calibration_analysis` parameter from fixture (not in current schema at `src/glassalpha/pipeline/audit.py:39-64`)

---

## ⚠️ P1 Fixes Required (macOS Only)

### 4. Missing System Dependencies on macOS

**Status:** ❌ NOT FIXED (requires workflow update)
**Files:** `.github/workflows/ci.yml` and `.github/workflows/determinism.yml`

**Errors:**

- WeasyPrint: `cannot load library 'libgobject-2.0-0'`
- XGBoost/LightGBM: `Library not loaded: @rpath/libomp.dylib`

**Recommended fix:**

```yaml
# Add to .github/workflows/ci.yml before test steps

- name: Install macOS system dependencies (macOS only)
  if: runner.os == 'macOS'
  run: |
    brew install gobject-introspection cairo pango gdk-pixbuf libffi
    brew install libomp
```

**Why:** GitHub macOS runners don't include these system libraries by default. Linux runners have them pre-installed.

---

## 🔧 P2 Fixes Required (Determinism Workflow)

### 5. Missing Test File Reference

**Status:** ❌ NOT FIXED
**File:** `.github/workflows/determinism.yml`

**Error:**

```
ERROR: file or directory not found:
tests/test_critical_regression_guards.py::TestCriticalRegressions::test_cli_determinism_regression_guard
```

**Options:**

1. Create missing test file
2. Update workflow to skip this specific test
3. Remove test reference from workflow

### 6. Missing Module Import

**Status:** ❌ NOT FIXED
**File:** `.github/workflows/determinism.yml`

**Error:**

```
ModuleNotFoundError: No module named 'glassalpha.explain.registry'
```

**Fix:** Update import in determinism test workflow (module was moved/renamed)

---

## Verification Commands

```bash
# Test P0 fixes locally
pytest tests/integration/test_pdf_generation_linux.py::TestPDFGenerationLinux::test_pdf_with_minimal_sections -v

# Test SBOM generation
cyclonedx-py environment --format json --outfile /tmp/test-sbom.json

# Verify template syntax
python -c "from jinja2 import Environment, FileSystemLoader; env = Environment(loader=FileSystemLoader('src/glassalpha/report/templates')); env.get_template('standard_audit.html')"
```

---

## Impact Analysis

**CI/CD Workflow:** Should pass after P0 fixes (except macOS-specific failures from P1)
**Tests with Environment Sync:** Should pass after P0 fixes (Linux), macOS needs P1
**Determinism Validation:** Requires P2 fixes for full pass

**Recommendation:**

1. Commit P0 fixes immediately (blocks everything)
2. Apply P1 fixes in follow-up PR (macOS-specific)
3. Evaluate P2 determinism test strategy separately

---

## Commit Message

```
fix(ci): Fix SBOM generation, template rendering, and test fixtures

- Fix cyclonedx-py command syntax (add 'environment' subcommand)
- Fix template manifest dict access (use .get() instead of attribute access)
- Remove obsolete calibration_analysis parameter from test fixtures

Fixes #<issue-number>
```
