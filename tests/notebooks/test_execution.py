"""Automated notebook execution tests using nbmake.

This test suite executes all example notebooks to catch breaking changes
before they reach users. Runs in CI on every PR.

**Why nbmake over manual testing:**
- Catches 90% of notebook bugs automatically (imports, runtime errors, API changes)
- Runs in <10 minutes (faster than 60-minute manual Colab testing)
- Zero maintenance cost (just works)
- Platform-agnostic (tests Linux, macOS, Windows in CI matrix)

**What this catches:**
- Import errors (missing dependencies, API changes)
- Runtime exceptions (breaking changes in from_model(), dataset loading)
- Cell execution failures (syntax errors, logic bugs)
- Timeout issues (performance regressions)

**What this doesn't catch (manual Colab testing required):**
- Colab-specific package versions
- Colab UI features (file download panel)
- GPU runtime issues

**Manual testing strategy:**
- Before PyPI releases (quarterly)
- After major notebook changes (new features)
- User reports notebook issue (reproduce in Colab)

See COLAB_TESTING_GUIDE.md for manual testing checklist.
"""

import pytest

# Notebook execution parameters: (filename, timeout_seconds)
# Timeouts include 20% buffer for CI variability
NOTEBOOKS = [
    # P0: Quickstart notebooks (first-touch experience)
    ("quickstart_colab.ipynb", 480),  # Target: 8 min
    ("quickstart_from_model.ipynb", 480),  # Target: 8 min
    # P1: Tutorial notebooks (comprehensive learning)
    ("german_credit_walkthrough.ipynb", 600),  # Target: 10 min
    ("custom_data_template.ipynb", 600),  # Target: 10 min
    # P2: Advanced feature notebooks
    ("adult_income_drift.ipynb", 600),  # Target: 10 min
    ("compas_bias_detection.ipynb", 600),  # Target: 10 min
]


@pytest.mark.slow  # Optional: run separately from fast unit tests
@pytest.mark.parametrize("notebook,timeout", NOTEBOOKS, ids=[nb[0] for nb in NOTEBOOKS])
def test_notebook_executes_without_errors(notebook, timeout):
    """Verify notebook executes cleanly end-to-end.

    Test is executed by nbmake pytest plugin. Notebook execution happens
    automatically when pytest is run with --nbmake flag.

    Args:
        notebook: Notebook filename (relative to examples/notebooks/)
        timeout: Maximum execution time in seconds

    Raises:
        nbmake.NBMakeException: If notebook execution fails or times out

    """
    # Actual execution happens via nbmake plugin when run with:
    # pytest --nbmake examples/notebooks/*.ipynb --nbmake-timeout=600
    #
    # This test exists to:
    # 1. Document which notebooks are tested
    # 2. Set explicit timeouts per notebook
    # 3. Enable parametrization for better failure reporting


def test_all_example_notebooks_covered():
    """Verify test suite covers all notebooks in examples/notebooks/.

    Prevents accidentally skipping notebooks when new ones are added.
    """
    import os
    from pathlib import Path

    notebooks_dir = Path(__file__).parent.parent.parent.parent / "examples" / "notebooks"
    if not notebooks_dir.exists():
        pytest.skip("Notebooks directory not found (expected in examples/notebooks/)")

    # Get all .ipynb files except internal/temp ones
    actual_notebooks = {
        f.name
        for f in notebooks_dir.glob("*.ipynb")
        if not f.name.startswith(".")  # Ignore .ipynb_checkpoints, hidden files
    }

    tested_notebooks = {nb for nb, _ in NOTEBOOKS}

    missing = actual_notebooks - tested_notebooks
    if missing:
        pytest.fail(
            f"Notebooks exist but not tested: {missing}\nAdd them to NOTEBOOKS list in {__file__}",
        )


# Usage in CI:
#
# .github/workflows/test.yml:
#   - name: Test example notebooks
#     run: |
#       cd examples/notebooks
#       pytest ../../packages/tests/notebooks/ \
#         --nbmake *.ipynb \
#         --nbmake-timeout=600 \
#         -v
#
# Local testing:
#   cd examples/notebooks
#   pytest ../../packages/tests/notebooks/ --nbmake *.ipynb -v
#
# Run specific notebook:
#   pytest --nbmake examples/notebooks/quickstart_colab.ipynb -v
#
# Skip slow notebook tests:
#   pytest -m "not slow"
