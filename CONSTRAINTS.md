# Dependency Constraints Management

This document describes how to maintain and update the dependency constraint files used for reproducible CI builds.

## Overview

GlassAlpha uses platform- and Python-version-specific constraint files to ensure:

- **Reproducible builds**: Same versions across all CI runs
- **Wheel-only installs**: No source builds that could fail or be slow
- **Platform compatibility**: Different constraints for Ubuntu, macOS, Windows

## Constraint File Structure

```
packages/constraints/
├── tooling-ubuntu-latest-py3.11.txt    # pip, setuptools, build, wheel
├── tooling-ubuntu-latest-py3.12.txt
├── tooling-macos-14-py3.11.txt
├── tooling-macos-14-py3.12.txt
├── constraints-ubuntu-latest-py3.11.txt  # All project + dev dependencies
├── constraints-ubuntu-latest-py3.12.txt
├── constraints-macos-14-py3.11.txt
└── constraints-macos-14-py3.12.txt
```

## Updating Constraints with pip-tools

### Prerequisites

```bash
# Install pip-tools
pipx install pip-tools
```

### Generating Constraints

**For test dependencies (explain + viz + dev, no docs):**

```bash
cd packages

# Ubuntu Python 3.11
pip-compile -o constraints/constraints-ubuntu-latest-py3.11.txt \
  --python-version 3.11 \
  --resolver backtracking \
  --strip-extras \
  --extra explain \
  --extra viz \
  --extra dev \
  pyproject.toml

# Ubuntu Python 3.12
pip-compile -o constraints/constraints-ubuntu-latest-py3.12.txt \
  --python-version 3.12 \
  --resolver backtracking \
  --strip-extras \
  --extra explain \
  --extra viz \
  --extra dev \
  pyproject.toml

# macOS Python 3.11
pip-compile -o constraints/constraints-macos-14-py3.11.txt \
  --python-version 3.11 \
  --resolver backtracking \
  --strip-extras \
  --extra explain \
  --extra viz \
  --extra dev \
  pyproject.toml

# macOS Python 3.12
pip-compile -o constraints/constraints-macos-14-py3.12.txt \
  --python-version 3.12 \
  --resolver backtracking \
  --strip-extras \
  --extra explain \
  --extra viz \
  --extra dev \
  pyproject.toml
```

**For tooling (build tools only):**

```bash
# Create a temporary requirements file
cat > /tmp/tooling.txt << 'EOF'
pip>=25,<26
setuptools>=75,<76
build>=1.2,<2
wheel>=0.45,<0.46
EOF

# Generate tooling constraints for each platform/Python combo
pip-compile -o constraints/tooling-ubuntu-latest-py3.11.txt \
  --python-version 3.11 \
  --resolver backtracking \
  /tmp/tooling.txt

pip-compile -o constraints/tooling-ubuntu-latest-py3.12.txt \
  --python-version 3.12 \
  --resolver backtracking \
  /tmp/tooling.txt

pip-compile -o constraints/tooling-macos-14-py3.11.txt \
  --python-version 3.11 \
  --resolver backtracking \
  /tmp/tooling.txt

pip-compile -o constraints/tooling-macos-14-py3.12.txt \
  --python-version 3.12 \
  --resolver backtracking \
  /tmp/tooling.txt
```

### Manual Additions

After generating with pip-tools, you may need to manually add:

1. **WeasyPrint** (only for Linux constraints where PDF tests run):

   ```bash
   echo "weasyprint==63.1" >> constraints/constraints-ubuntu-latest-py3.11.txt
   echo "weasyprint==63.1" >> constraints/constraints-ubuntu-latest-py3.12.txt
   ```

   Note: WeasyPrint uses major.minor versioning (63.0, 63.1, 64.0, etc.), not patch versions.

2. **Platform-specific pins** that pip-tools might miss

## Why Not Include docs Extra?

PDF generation (the `docs` extra with WeasyPrint) is only tested on Linux in CI to avoid:

- Long system dependency install times on macOS (brew install cairo, pango, etc.)
- Dependency version skew across platforms
- macOS-specific issues with cairo/pango

WeasyPrint is added manually to Linux constraint files only.

## Verification After Updates

After generating new constraints, verify they work:

```bash
cd packages

# Clean environment
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate

# Install with constraints
pip install -r constraints/tooling-ubuntu-latest-py3.12.txt
pip install -e ".[explain,viz,dev]" -c constraints/constraints-ubuntu-latest-py3.12.txt

# Enforce wheel-only for heavy dependencies
pip install --only-binary=:all: \
  numpy scipy pandas scikit-learn shap xgboost lightgbm matplotlib seaborn \
  -c constraints/constraints-ubuntu-latest-py3.12.txt

# Verify no conflicts
pip check

# Run tests
pytest -x
```

## When to Update Constraints

1. **Adding new dependencies** to `pyproject.toml`
2. **Security updates** for pinned versions
3. **Quarterly maintenance** to stay current with upstream
4. **After failed CI builds** due to version conflicts

## Pinning Strategy

- **Core dependencies**: Pin major+minor (e.g., `numpy==2.3.3`)
- **Build tools**: Pin major+minor (e.g., `pip==25.0`)
- **WeasyPrint**: Pin major only (e.g., `>=63,<64` in pyproject.toml, `==63.1` in constraints)
- **Test tools**: Pin major+minor (e.g., `pytest==8.4.2`)

## Troubleshooting

### Dependency Conflict During Constraint Generation

If `pip-compile` reports conflicts:

1. Check `pyproject.toml` version ranges aren't too restrictive
2. Try without `--strip-extras` to see full dependency tree
3. Update the conflicting package's version range in `pyproject.toml`
4. Regenerate constraints

### Platform-Specific Failures

If a package fails on one platform but not others:

1. Check if wheels are available for that platform/Python combo on PyPI
2. Consider making the dependency optional for that platform
3. Document platform-specific constraints in this file

### Source Builds in CI

If CI shows "Building wheel for X":

1. Find the version that has wheels on PyPI
2. Pin that version in the constraint file
3. Verify with `pip install --only-binary=:all:` locally

## Automated Updates

Consider setting up:

- **Renovate** to auto-create PRs for dependency updates
- **Dependabot** as an alternative
- **GitHub Actions** workflow to regenerate constraints weekly

Example Renovate config for constraint files:

```json
{
  "pip_compile": {
    "enabled": true,
    "fileMatch": ["constraints/.*\\.txt$"]
  }
}
```

## Related Files

- `pyproject.toml` - Source of truth for dependency ranges
- `packages/constraints.txt` - Legacy unified constraint file (deprecated)
- `.github/workflows/ci.yml` - CI workflow that uses these constraints
- `CONSTRAINTS.md` - This file
