# Contributing to GlassAlpha

## Development Environment Setup

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/glassalpha.git
cd glassalpha

# 2. Set up development environment (one command does everything)
make dev-setup
```

That's it! This single command:

- Creates a virtual environment (`.venv/`)
- Installs the package in **editable mode** with all dependencies
- Installs git hooks (pre-commit and pre-push)
- Runs environment validation (`glassalpha doctor`)

### Why Editable Mode Matters

**Editable mode** (`pip install -e .`) is crucial because:

- Changes to source code are immediately active (no reinstall needed)
- Tests run against your current code, not a stale installed version
- Your environment stays in sync with the repository

**Common Problem**: Running tests with a stale installed version leads to confusing failures where:

- Tests pass locally but fail in CI
- Changes you make don't affect test results
- Different developers see different behavior

### Environment Synchronization

GlassAlpha uses a **strict environment synchronization system** to ensure:

1. Your local venv matches the source code
2. CI uses the same versions
3. All contributors have consistent environments

#### Check Your Environment

Before running tests:

```bash
make check-venv
```

This validates:

- ✓ Virtual environment exists (`.venv/`)
- ✓ Package is installed in editable mode
- ✓ You're using the venv's Python (warns if not activated)

#### Fix Environment Issues

If `make check-venv` shows problems:

```bash
# Quick fix: reinstall in editable mode
.venv/bin/pip install -e . --no-deps

# Full reset: recreate environment
make sync-deps
```

### Testing

```bash
# Run full test suite (automatically checks venv first)
make test

# If environment is out of sync, you'll see:
#   ❌ Environment out of sync - package not in editable mode
#   💡 Quick fix: Run one of these commands:
#      make sync-deps           # Recommended: full sync
#      make test AUTO_FIX=1     # Auto-fix and continue
#      .venv/bin/pip install -e . --no-deps  # Manual fix

# Option 1: Fix manually then re-run
make sync-deps
make test

# Option 2: Auto-fix in one command
make test AUTO_FIX=1

# Run specific tests
.venv/bin/python3 -m pytest tests/test_specific.py -v

# Run with coverage
make coverage
```

**How it works**:

1. `make test` automatically runs `check-venv` first
2. If out of sync, it **fails** with clear fix instructions
3. You either run `make sync-deps` or use `AUTO_FIX=1` to auto-fix
4. Tests won't run until environment is valid

**Important**: Always use `.venv/bin/python3` or activate the venv:

```bash
source .venv/bin/activate  # Now 'python3' uses venv
make test                   # Uses venv automatically
```

### How Other Projects Handle This

GlassAlpha's approach combines best practices from major Python projects:

#### 1. **Django/Flask**: Requirements Files + Editable Install

```bash
pip install -e ".[dev]"      # Editable mode
pip install -r requirements/dev.txt  # Pin versions
```

#### 2. **pytest/numpy**: Constraints Files

```
constraints/
├── dev-requirements.txt     # Frozen development versions
├── ci-requirements.txt      # CI-specific pins
└── docs-requirements.txt    # Documentation builds
```

#### 3. **Our Approach**: Automated Validation

```bash
make test                    # Automatically checks venv
make check-venv              # Manual validation
make sync-deps               # Auto-sync environment
```

**Key Innovation**: We add a `check-venv` step that runs **before every test**, catching environment drift immediately instead of debugging mysterious failures later.

### Dependency Management

#### For Daily Development

```bash
# Just keep your environment in sync
.venv/bin/pip install -e . --no-deps  # After pulling changes
```

#### For Maintainers: Updating Dependencies

When adding or updating dependencies:

```bash
# 1. Update pyproject.toml
# 2. Reinstall
.venv/bin/pip install -e ".[dev,all]"

# 3. Freeze current versions
make freeze-deps

# 4. Commit both files
git add pyproject.toml constraints/dev-requirements.txt
git commit -m "Update dependencies: added X, upgraded Y"
```

This creates a **single source of truth** for dependency versions that:

- Works locally (via editable install)
- Works in CI (via constraints file)
- Works for contributors (via `make dev-setup`)

### Pre-commit Checks

Before committing:

```bash
make check
```

This runs:

- ✓ Smoke test (validates critical contracts)
- ✓ Workflow validation (YAML syntax + action versions)
- ✓ Sigstore check (signing process validation)
- ✓ Packaging check (MANIFEST.in completeness)
- ✓ Determinism check (CI compatibility)
- ✓ Environment check (`glassalpha doctor`)
- ✓ Documentation check (CLI docs up to date)

### Git Hooks

Installed automatically by `make dev-setup`:

**Pre-commit**: Runs smoke test only if fragile areas changed
**Pre-push**: Always runs smoke test to catch regressions

### CI Alignment

**Q: How does CI handle environment sync?**

**A: CI automatically validates on every run:**

```yaml
# In .github/workflows/test-with-sync.yml
- name: Install dependencies
  run: pip install -e ".[dev,all]" # Always editable mode

- name: Validate environment sync
  run: python scripts/sync-deps.py --check # Fails if out of sync
```

**Key differences**:

- **Local**: You might have stale installs → `make test` catches it
- **CI**: Fresh install every time → Should always be in sync

**Q: What if CI environment is out of sync?**

**A: CI will FAIL with clear error:**

```
❌ Environment out of sync in CI
This should not happen - CI installs in editable mode by default
```

This means there's a bug in the CI workflow (not your code). File an issue!

**Q: Do I need to manually update CI?**

**A: No! CI auto-syncs on every run:**

1. Fresh environment for every CI run
2. `pip install -e ".[dev,all]"` → always installs in editable mode
3. `scripts/sync-deps.py --check` → validates it worked
4. If validation fails → CI fails (bug in workflow, not your code)

Your local environment matches CI by:

1. **Same Python version**: 3.11+ (check with `python3 --version`)
2. **Same install method**: Editable mode (`pip install -e .`)
3. **Same checks**: `make check` runs the same validations as CI
4. **Same test command**: `make test` uses same pytest invocation
5. **Same validation**: Both run `scripts/sync-deps.py --check`

### Troubleshooting

#### "Tests pass locally but fail in CI"

**Cause**: Environment mismatch (stale installed version)

**Fix**:

```bash
make check-venv  # Diagnose the issue
.venv/bin/pip install -e . --no-deps  # Sync with source
make test        # Re-run tests
```

#### "ImportError: cannot import name X"

**Cause**: Package not installed or old version cached

**Fix**:

```bash
.venv/bin/pip install -e . --force-reinstall --no-deps
```

#### "Tests run but changes don't take effect"

**Cause**: Not using venv's Python

**Fix**:

```bash
source .venv/bin/activate  # Or use .venv/bin/python3 directly
```

### Best Practices

1. **Always activate venv** or use `.venv/bin/python3` explicitly
2. **Run `make check-venv`** after pulling changes
3. **Run `make check`** before committing
4. **Run `make test`** before pushing
5. **Use `make sync-deps`** if environment seems broken

### Why This System?

**Problem**: Developers waste hours debugging issues caused by:

- Stale installed packages
- Different Python versions
- Missing dependencies
- Environment drift

**Solution**: Automated validation that catches these issues **immediately** with clear fix instructions.

**Result**:

- ✅ Tests are reliable
- ✅ CI matches local
- ✅ New contributors onboard fast
- ✅ No mysterious environment bugs

### Questions?

If you encounter issues with the development environment:

1. Run `make check-venv` to diagnose
2. Try `make sync-deps` to reset
3. Check this guide for common solutions
4. Open an issue with the output of `glassalpha doctor`
