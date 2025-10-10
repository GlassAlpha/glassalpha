# Notebook Execution Tests

Automated testing of example notebooks using [nbmake](https://github.com/treebeardtech/nbmake).

## Quick Start

```bash
# Test all notebooks
cd examples/notebooks
pytest ../../packages/tests/notebooks/ --nbmake *.ipynb -v

# Test specific notebook
pytest --nbmake examples/notebooks/quickstart_colab.ipynb -v

# Test with custom timeout
pytest --nbmake examples/notebooks/*.ipynb --nbmake-timeout=300 -v
```

## What This Tests

- ✅ **Import errors**: Catches missing dependencies, API changes
- ✅ **Runtime exceptions**: Detects breaking changes in `from_model()`
- ✅ **Cell execution**: Validates syntax, logic, data loading
- ✅ **Timeout issues**: Detects performance regressions

## What This Doesn't Test

- ⚠️ **Colab-specific environment**: Python versions, preinstalled packages
- ⚠️ **Colab UI features**: File download panel, GPU runtime
- ⚠️ **Output formatting**: nbmake tests execution only, not visual output

For Colab-specific testing, see `examples/notebooks/COLAB_TESTING_GUIDE.md`.

## CI Integration

Runs automatically in CI on every PR:

```yaml
# In packages/ci-template.yml
- name: Test example notebooks
  run: |
    cd examples/notebooks
    pytest ../../packages/tests/notebooks/ \
      --nbmake *.ipynb \
      --nbmake-timeout=600 \
      -v
```

## Adding New Notebooks

When adding a new notebook to `examples/notebooks/`:

1. Add entry to `NOTEBOOKS` list in `test_execution.py`:

   ```python
   NOTEBOOKS = [
       ("new_notebook.ipynb", 600),  # timeout in seconds
   ]
   ```

2. Test locally before pushing:

   ```bash
   pytest --nbmake examples/notebooks/new_notebook.ipynb -v
   ```

3. The test `test_all_example_notebooks_covered()` will fail if you forget to add it.

## Troubleshooting

### Notebook test fails locally but works in Colab

This is expected! nbmake tests in your local environment, which may differ from Colab:

- Different Python version
- Different package versions
- Different default settings

**Solution**: Either fix the notebook to work in both environments, or document Colab-specific requirements.

### Timeout errors

Increase timeout in `NOTEBOOKS` list:

```python
("slow_notebook.ipynb", 1200),  # 20 minutes
```

### Import errors

Ensure dependencies are installed:

```bash
pip install -e ".[explain]"  # For notebooks using SHAP, XGBoost
```

## Performance

Typical execution times (on local machine):

- `quickstart_from_model.ipynb`: ~2 seconds
- `quickstart_colab.ipynb`: ~30 seconds
- `german_credit_walkthrough.ipynb`: ~45 seconds
- `adult_income_drift.ipynb`: ~60 seconds

**Total CI time**: ~10 minutes for all notebooks (includes setup overhead)

## Test Coverage

| Notebook                        | Priority | Status | Last Tested |
| ------------------------------- | -------- | ------ | ----------- |
| quickstart_colab.ipynb          | P0       | ✅     | Every PR    |
| quickstart_from_model.ipynb     | P0       | ✅     | Every PR    |
| german_credit_walkthrough.ipynb | P1       | ✅     | Every PR    |
| custom_data_template.ipynb      | P1       | ✅     | Every PR    |
| adult_income_drift.ipynb        | P2       | ✅     | Every PR    |
| compas_bias_detection.ipynb     | P2       | ✅     | Every PR    |

## Manual Colab Testing

Automated tests catch 90% of issues. For the remaining 10%, use manual Colab testing:

**When to manually test:**

- Before PyPI releases (quarterly)
- After major notebook changes (new features)
- User reports notebook issue (reproduce in Colab)

See `examples/notebooks/COLAB_TESTING_GUIDE.md` for checklist.

## Related Documentation

- [COLAB_TESTING_GUIDE.md](../../../examples/notebooks/COLAB_TESTING_GUIDE.md) - Manual testing checklist
- [NOTEBOOK_FIXES_SUMMARY.md](../../../examples/notebooks/NOTEBOOK_FIXES_SUMMARY.md) - Recent bug fixes
- [nbmake documentation](https://github.com/treebeardtech/nbmake) - Official docs
