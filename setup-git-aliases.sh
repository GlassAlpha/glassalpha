#!/bin/bash
# ============================================================================
# Git Aliases Setup Script (ROOT FILE - DO NOT MOVE)
# ============================================================================
# INTENTIONAL LOCATION: This file lives in the repository root
# WHY: Developer convenience for local git workflow setup
# NOT DISTRIBUTED: Excluded from PyPI package via MANIFEST.in (line 70)
# PURPOSE: Creates local git aliases for linting and safe commits
# USAGE: Run once after cloning: `bash setup-git-aliases.sh`
# ============================================================================
# Set up Git aliases to prevent pre-commit hook failures

echo "🔧 Setting up Git aliases for GlassAlpha..."

# Add git alias for safe commits
git config alias.safe-commit '!f() {
    echo "Running pre-commit checks...";
    source venv/bin/activate && ruff check src/ tests/ --fix && ruff format src/ tests/ && pre-commit run --all-files &&
    git add . && git commit -m "$1";
}; f'

# Add alias for just linting (no commit)
git config alias.lint '!f() {
    source venv/bin/activate && ruff check src/ tests/ --fix && ruff format src/ tests/ && pre-commit run --all-files;
}; f'

# Add alias for quick fix
git config alias.fix '!f() {
    source venv/bin/activate && ruff check src/ tests/ --fix && ruff format src/ tests/;
}; f'

echo ""
echo "Git aliases created! You can now use:"
echo ""
echo "   git fix              - Quick auto-fix with Ruff"
echo "   git lint             - Full pre-commit check (no commit)"
echo "   git safe-commit 'msg' - Lint + auto-fix + commit in one command"
echo ""
echo "Example:"
echo "   git safe-commit 'Add new feature'"
echo ""

chmod +x setup-git-aliases.sh
