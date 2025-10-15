# GlassAlpha - Wheel-first development workflow
.PHONY: smoke build install test lint clean clean-temp hooks help dev-setup check check-fast check-workflows check-sigstore check-packaging check-determinism check-notebooks check-venv sync-deps freeze-deps

# Default target
help:
	@echo "GlassAlpha Development Commands:"
	@echo ""
	@echo "🚀 dev-setup  - Complete dev environment setup (one-time)"
	@echo "🔥 smoke      - Run wheel smoke test (validates 4 critical contracts)"
	@echo "⚡ check-fast - Fast validation (smoke + workflows + packaging + docs, ~30s)"
	@echo "🔍 check      - Full pre-commit check (includes PDF determinism, ~2-3min)"
	@echo "📦 build      - Build wheel"
	@echo "📥 install    - Install wheel (for testing)"
	@echo "🧪 test       - Run full test suite (auto-checks venv first)"
	@echo "📊 coverage   - Run tests with coverage report (terminal + HTML)"
	@echo "🔍 lint       - Run linting"
	@echo "🧹 clean      - Clean build artifacts"
	@echo "🧹 clean-temp - Remove AI-generated test files from root (auto-runs before tests)"
	@echo "🪝 hooks      - Install git hooks (pre-commit + pre-push)"
	@echo "🔐 check-sigstore - Test sigstore signing process locally"
	@echo "🔍 check-workflows - Validate GitHub Actions workflows"
	@echo "📦 check-packaging - Validate MANIFEST.in and pyproject.toml"
	@echo "📓 check-notebooks - Validate example notebooks (imports, structure)"
	@echo "📚 check-docs - Validate CLI documentation (auto-fixes if stale)"
	@echo ""
	@echo "Environment Management:"
	@echo "🔍 check-venv - Check if venv is in sync with source code"
	@echo "🔄 sync-deps  - Auto-fix environment (reinstall in editable mode)"
	@echo "🧊 freeze-deps - Freeze current versions to constraints file"
	@echo ""
	@echo "Getting Started:"
	@echo "1. make dev-setup      (one-time: installs deps + hooks + runs doctor)"
	@echo "2. make check          (before commit: validates setup + packaging)"
	@echo "3. git commit          (pre-commit: smoke test if fragile areas changed)"
	@echo "4. git push            (pre-push: always runs smoke test)"
	@echo ""
	@echo "Environment Troubleshooting:"
	@echo "• Tests failing mysteriously? → make check-venv"
	@echo "• Environment out of sync? → make sync-deps"
	@echo "• Quick auto-fix? → make test AUTO_FIX=1"

# Smoke test - the key guardrail against CI thrashing
smoke:
	@echo "🔥 Running wheel smoke test..."
	./scripts/wheel_smoke.sh

# Build wheel
build:
	python3 -m build

# Install built wheel
install: build
	python3 -m pip install --force-reinstall --no-deps dist/*.whl

# Test suite (requires dependencies)
# Note: Parallel execution with pytest-xdist speeds up tests but some tests
# (subprocess-based, shared file writes) must be marked to run serially
# Automatically cleans AI-generated test outputs before running
# Pre-test validation - ensures environment, packaging, and contracts work before tests
check-before-test: check-venv smoke check-packaging check-determinism
	@echo ""
	@echo "🏥 Checking environment..."
	@glassalpha doctor

test: check-before-test clean-temp
	python3 -m pytest -q

# Test with coverage report
coverage: clean-temp
	@echo "🔍 Running tests with coverage..."
	pytest --cov=glassalpha --cov-report=term-missing --cov-report=html -q --disable-warnings
	@echo ""
	@echo "✅ Coverage report generated:"
	@echo "   Terminal: See output above"
	@echo "   HTML: Open htmlcov/index.html in browser"
	@echo ""
	@echo "📊 Running coverage gate..."
	python .ci/coverage_gate.py

# Linting
lint:
	ruff check .
	mypy src/

# Clean build artifacts only
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Clean AI-generated test files from root (runs automatically before tests)
clean-temp:
	@./scripts/cleanup-temp-files.sh execute 2>&1 | grep -v "^Would delete" || true

# Install git hooks (one-time setup)
hooks:
	@echo "🪝 Installing git hooks..."
	@mkdir -p .git/hooks
	@ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
	@ln -sf ../../scripts/pre-push .git/hooks/pre-push
	@echo "   ✅ pre-commit hook: runs smoke test when fragile areas change"
	@echo "   ✅ pre-push hook: always runs smoke test before push"
	@echo ""
	@echo "Hooks installed! Your commits and pushes are now protected."

# Complete dev environment setup (one-time)
dev-setup:
	@echo "🚀 Setting up development environment..."
	@echo ""
	@echo "📦 Installing package with all dependencies..."
	pip install -e ".[dev,all]"
	@echo ""
	@echo "🪝 Installing git hooks..."
	@$(MAKE) hooks
	@echo ""
	@echo "🏥 Running environment check..."
	@glassalpha doctor
	@echo ""
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make check' before committing (includes workflow + sigstore + determinism validation)"
	@echo "  - Run 'make test' for full test suite"
	@echo "  - Run 'glassalpha audit --config quickstart.yaml --output test.pdf' for a quick test"

# Fast validation loop (skip expensive PDF tests)
check-fast: smoke check-workflows check-packaging check-notebooks check-docs
	@echo ""
	@echo "🏥 Checking environment..."
	@glassalpha doctor
	@echo ""
	@echo "✅ Fast checks passed!"

# Full pre-commit check (includes PDF determinism)
check: clean-temp smoke check-workflows check-sigstore check-packaging check-determinism
	@echo ""
	@echo "🏥 Checking environment..."
	@glassalpha doctor
	@echo ""
	@echo "📚 Checking documentation..."
	@$(MAKE) check-docs
	@echo ""
	@echo "✅ All checks passed - ready to commit!"

# Check CLI documentation is current (auto-fix if stale)
check-docs:
	@echo "🔍 Checking CLI documentation..."
	@if ! python3 scripts/generate_cli_docs.py --check 2>/dev/null; then \
		echo "   🔧 Auto-fixing outdated CLI documentation..."; \
		python3 scripts/generate_cli_docs.py --output site/docs/reference/cli.md; \
		echo "   ✓ CLI documentation regenerated"; \
	fi

# Validate example notebooks
check-notebooks:
	@echo "📓 Validating example notebooks..."
	@python3 scripts/validate_notebooks.py

# Test sigstore signing process locally
check-sigstore:
	@echo "🔐 Testing sigstore signing process locally..."
	@./scripts/test_sigstore_local.sh

# Validate GitHub Actions workflows
check-workflows:
	@echo "🔍 Validating GitHub Actions workflows..."
	@echo ""
	@echo "🔧 Auto-fixing YAML formatting..."
	@if command -v prettier >/dev/null 2>&1; then \
		prettier --write .github/workflows/*.yml 2>/dev/null || true; \
		echo "   ✓ YAML files formatted with prettier"; \
	else \
		echo "   ⚠️  prettier not installed, skipping auto-fix"; \
		echo "   💡 Install with: npm install -g prettier"; \
	fi
	@echo ""
	@echo "📋 Checking workflow YAML syntax..."
	@for file in .github/workflows/*.yml .github/workflows/*.yaml; do \
		if [ -f "$$file" ]; then \
			echo "   Checking $$file..."; \
			if command -v yamllint >/dev/null 2>&1; then \
				yamllint --config-file .yamllint "$$file" || exit 1; \
			else \
				python3 -c "import yaml; yaml.safe_load(open('$$file'))" || (echo "❌ YAML syntax error in $$file"; exit 1); \
			fi; \
		fi; \
	done
	@echo ""
	@echo "🔗 Checking action versions and availability..."
	@echo "   (This checks common actions - may need internet connection)"
	@python3 scripts/validate_workflows.py

# Validate packaging (MANIFEST.in + pyproject.toml)
check-packaging:
	@echo "📦 Validating packaging configuration..."
	@echo ""
	@echo "🔍 Checking pyproject.toml schema..."
	@if command -v validate-pyproject >/dev/null 2>&1; then \
		validate-pyproject pyproject.toml || exit 1; \
	else \
		echo "   ⚠️  validate-pyproject not installed, skipping schema validation"; \
	fi
	@echo ""
	@echo "📦 Checking MANIFEST.in completeness..."
	@if command -v check-manifest >/dev/null 2>&1; then \
		check-manifest --verbose || exit 1; \
	else \
		echo "   ⚠️  check-manifest not installed, skipping manifest validation"; \
		echo "   💡 Install with: pip install check-manifest"; \
		echo "   📝 This validation runs automatically in CI"; \
	fi

# Validate determinism (matches CI requirements)
check-determinism:
	@echo "🔬 Validating determinism (CI compatibility check)..."
	@if [ -f "./scripts/test_determinism.sh" ]; then \
		if python3 -c "import shap, weasyprint" 2>/dev/null; then \
			./scripts/test_determinism.sh full || exit 1; \
		else \
			echo "   ⚠️  Optional determinism dependencies not available (shap, weasyprint)"; \
			echo "   💡 Run 'pip install -e \".[all]\"' to enable determinism validation"; \
			echo "   📝 Determinism validation runs automatically in CI"; \
		fi \
	else \
		echo "   ⚠️  Determinism test script not found, skipping"; \
		echo "   💡 This validation ensures local environment matches CI"; \
	fi

# Check that venv is in sync with source code (with auto-fix option)
check-venv:
	@echo "🔍 Checking virtual environment sync..."
	@if [ -d ".venv" ]; then \
		if [ -z "$$VIRTUAL_ENV" ]; then \
			echo "   ⚠️  Virtual environment exists but not activated"; \
			echo "   💡 Run: source .venv/bin/activate"; \
			echo "   📝 Or use: .venv/bin/python3 -m pytest"; \
		fi; \
		./.venv/bin/python3 -c "import glassalpha; import sys; from pathlib import Path; src_path = Path('src/glassalpha/__init__.py'); pkg_path = Path(glassalpha.__file__); is_editable = 'site-packages' not in str(pkg_path); print(f'   ✓ Package installed in editable mode' if is_editable else f'   ✗ Package NOT in editable mode: {pkg_path}'); sys.exit(0 if is_editable else 1)" || ( \
			echo ""; \
			echo "   ❌ Environment out of sync - package not in editable mode"; \
			echo ""; \
			if [ "$$AUTO_FIX" = "1" ]; then \
				echo "   🔧 Auto-fixing (set by AUTO_FIX=1)..."; \
				.venv/bin/pip install -e . --no-deps -q; \
				echo "   ✅ Fixed! Package reinstalled in editable mode"; \
			else \
				echo "   💡 Quick fix: Run one of these commands:"; \
				echo "      make sync-deps           # Recommended: full sync"; \
				echo "      make test AUTO_FIX=1     # Auto-fix and continue"; \
				echo "      .venv/bin/pip install -e . --no-deps  # Manual fix"; \
				exit 1; \
			fi \
		); \
	else \
		echo "   ❌ No .venv directory found"; \
		echo ""; \
		echo "   💡 Fix with: make dev-setup"; \
		exit 1; \
	fi
	@echo "   ✓ Virtual environment is properly configured"

# Synchronize development environment
sync-deps:
	@echo "🔄 Synchronizing development dependencies..."
	@python3 scripts/sync-deps.py --sync

# Freeze current dependencies to constraints file
freeze-deps:
	@echo "🧊 Freezing current dependencies..."
	@python3 scripts/sync-deps.py --freeze
