# New Chat Prompt: Fix Remaining Test Import Errors

I need to fix 6 test files that have config import errors after a major architecture cleanup.

## Context

We just completed a successful architecture simplification:

- ✅ Removed registry pattern (explicit dispatch now)
- ✅ Removed strict mode tests
- ✅ Removed `audit_profile` field
- ✅ **145+ core tests passing** (config, explainer, metrics, XGBoost, model integration)

**Current blocker:** 18 collection errors from 6 test files trying to import from `glassalpha.config` package instead of the `config.py` module.

## The Problem

**These 6 files fail with `ImportError: cannot import name 'AuditConfig' from 'glassalpha.config'`:**

1. `tests/unit/test_pdf_rendering.py`
2. `tests/integration/test_preprocessing_e2e.py`
3. `tests/test_end_to_end.py`
4. `tests/test_ci_regression_guards.py`
5. `tests/test_golden_report_snapshots.py`
6. `tests/preprocessing/test_strict_mode.py`

**Root cause:** Config classes are defined in `src/glassalpha/config.py` (module file), but tests try to import from `src/glassalpha/config/` (package directory). The package `__init__.py` only exports `builder`, not the config classes.

## The Solution (Proven Pattern)

**Working pattern from `tests/config/test_config.py` (37/37 tests passing):**

```python
import sys
from pathlib import Path
import importlib.util

# Add src to path and import from the main config file
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from the config module file
spec = importlib.util.spec_from_file_location("config", src_path / "glassalpha" / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Import classes and functions
AuditConfig = config_module.AuditConfig
DataConfig = config_module.DataConfig
ExplainerConfig = config_module.ExplainerConfig
ModelConfig = config_module.ModelConfig
load_config_from_file = config_module.load_config_from_file
load_yaml = config_module.load_yaml
```

**Why this works:** Avoids circular import between package and module, loads directly from file.

## Your Task

1. **Find problematic imports** in each of the 6 files:

   ```python
   from glassalpha.config import AuditConfig, DataConfig, ...
   ```

2. **Replace with working pattern** (adjust `parent` depth based on file location):

   - `tests/*.py` → `parent.parent / "src"`
   - `tests/unit/*.py` → `parent.parent.parent / "src"`
   - `tests/integration/*.py` → `parent.parent.parent / "src"`
   - `tests/preprocessing/*.py` → `parent.parent.parent / "src"`

3. **Test each fix:**

   ```bash
   pytest tests/unit/test_pdf_rendering.py --tb=no -q
   ```

4. **Verify all collection errors resolved:**
   ```bash
   pytest tests/ --co -q 2>&1 | grep "ERROR" | wc -l
   # Should be 0
   ```

## Expected Outcome

- Zero collection errors
- 200+ tests passing
- All core audit functionality verified

## Quick Reference

```bash
# Find files to fix
find tests/ -name "*.py" -exec grep -l "from glassalpha.config import AuditConfig" {} \;

# Test progress
pytest tests/ --tb=no -q 2>&1 | tail -1
```

## Important Notes

- ✅ The architecture is solid - just apply the pattern
- ✅ Don't change `src/glassalpha/config/__init__.py` (circular import risk)
- ✅ This pattern is proven to work (37/37 config tests passing)
- ✅ Full context available in `HANDOFF_NEXT_CHAT.md`

Ready to knock out these 6 files!
