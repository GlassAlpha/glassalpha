# Regression Tests for GlassAlpha

This directory contains high-signal regression tests that catch the exact contract violations that have caused CI failures.

## Critical Regression Tests

### `test_feature_alignment_contract.py`
- **Purpose**: Prevents column-drift handling regressions
- **Contract**: Same width + renamed columns → accept positionally; otherwise reindex with fill_value=0
- **Catches**: `ValueError: The feature names should match...` from sklearn

### `test_logging_no_printf.py`
- **Purpose**: Prevents printf-style logging regressions
- **Contract**: All logger calls must use single positional argument (f-strings)
- **Catches**: `logger.info("... %s", thing)` patterns that break test mocks

### `test_constants_contract.py`
- **Purpose**: Prevents contract string drift and packaging issues
- **Contract**: Exact error messages and template resource availability
- **Catches**: String changes that break error handling and missing wheel resources

## Integration

These tests run automatically with:
```bash
pytest tests/test_*_contract.py
```

For belt-and-suspenders protection, consider adding to pre-commit hooks:
```bash
# Run just the regression tests on changed files
pytest tests/test_feature_alignment_contract.py tests/test_logging_no_printf.py tests/test_constants_contract.py
```

## Why These Tests Matter

Each test prevents a specific category of regressions that have caused:
- CI test failures due to column handling
- Mock assertion failures due to logging format changes
- Wheel packaging issues with missing resources
- Contract drift in error messages

Fast execution (~100ms total) makes them suitable for pre-commit hooks.
