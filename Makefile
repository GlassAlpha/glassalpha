# GlassAlpha - Wheel-first development workflow
.PHONY: smoke build install test lint clean hooks help dev-setup check check-workflows check-sigstore check-packaging check-determinism

# Default target
help:
	@echo "GlassAlpha Development Commands:"
	@echo ""
	@echo "🚀 dev-setup  - Complete dev environment setup (one-time)"
	@echo "🔥 smoke      - Run wheel smoke test (validates 4 critical contracts)"
	@echo "🔍 check      - Quick pre-commit check (smoke test + doctor + packaging + determinism)"
	@echo "📦 build      - Build wheel"
	@echo "📥 install    - Install wheel (for testing)"
	@echo "🧪 test       - Run full test suite"
	@echo "📊 coverage   - Run tests with coverage report (terminal + HTML)"
	@echo "🔍 lint       - Run linting"
	@echo "🧹 clean      - Clean build artifacts"
	@echo "🪝 hooks      - Install git hooks (pre-commit + pre-push)"
	@echo "🔐 check-sigstore - Test sigstore signing process locally"
	@echo "🔍 check-workflows - Validate GitHub Actions workflows"
	@echo "📦 check-packaging - Validate MANIFEST.in and pyproject.toml"
	@echo ""
	@echo "Getting Started:"
	@echo "1. make dev-setup      (one-time: installs deps + hooks + runs doctor)"
	@echo "2. make check          (before commit: validates setup + packaging)"
	@echo "3. git commit          (pre-commit: smoke test if fragile areas changed)"
	@echo "4. git push            (pre-push: always runs smoke test)"

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
test:
	python3 -m pytest -q

# Test with coverage report
coverage:
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

# Clean artifacts
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

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

# Fast pre-commit check
check: smoke check-workflows check-sigstore check-packaging check-determinism
	@echo ""
	@echo "🏥 Checking environment..."
	@glassalpha doctor
	@echo ""
	@echo "📚 Checking documentation..."
	@$(MAKE) check-docs
	@echo ""
	@echo "✅ All checks passed - ready to commit!"

# Check CLI documentation is current
check-docs:
	@echo "🔍 Checking CLI documentation..."
	@python scripts/generate_cli_docs.py --check

# Test sigstore signing process locally
check-sigstore:
	@echo "🔐 Testing sigstore signing process locally..."
	@./scripts/test_sigstore_local.sh

# Validate GitHub Actions workflows
check-workflows:
	@echo "🔍 Validating GitHub Actions workflows..."
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
	@if [ -f "./scripts/test_determinism_local.sh" ]; then \
		if python3 -c "import shap, weasyprint" 2>/dev/null; then \
			./scripts/test_determinism_local.sh || exit 1; \
		else \
			echo "   ⚠️  Optional determinism dependencies not available (shap, weasyprint)"; \
			echo "   💡 Run 'pip install -e \".[all]\"' to enable determinism validation"; \
			echo "   📝 Determinism validation runs automatically in CI"; \
		fi \
	else \
		echo "   ⚠️  Determinism test script not found, skipping"; \
		echo "   💡 This validation ensures local environment matches CI"; \
	fi
