# GlassAlpha - Wheel-first development workflow
.PHONY: smoke build install test lint clean hooks help dev-setup check

# Default target
help:
	@echo "GlassAlpha Development Commands:"
	@echo ""
	@echo "🚀 dev-setup  - Complete dev environment setup (one-time)"
	@echo "🔥 smoke      - Run wheel smoke test (validates 4 critical contracts)"
	@echo "🔍 check      - Quick pre-commit check (smoke test + doctor)"
	@echo "📦 build      - Build wheel"
	@echo "📥 install    - Install wheel (for testing)"
	@echo "🧪 test       - Run full test suite"
	@echo "📊 coverage   - Run tests with coverage report (terminal + HTML)"
	@echo "🔍 lint       - Run linting"
	@echo "🧹 clean      - Clean build artifacts"
	@echo "🪝 hooks      - Install git hooks (pre-commit + pre-push)"
	@echo ""
	@echo "Getting Started:"
	@echo "1. make dev-setup      (one-time: installs deps + hooks + runs doctor)"
	@echo "2. make check          (before commit: validates setup)"
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
# Use pytest-xdist for parallel execution and test isolation (prevents test pollution)
test:
	pytest -n auto -q

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
	@echo "  - Run 'make check' before committing"
	@echo "  - Run 'make test' for full test suite"
	@echo "  - Run 'glassalpha audit --config quickstart.yaml --output test.pdf' for a quick test"

# Fast pre-commit check
check: smoke
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
