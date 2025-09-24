#!/bin/bash
# Set up Git aliases to prevent pre-commit hook failures

echo "🔧 Setting up Git aliases for Glass Alpha..."

# Add git alias for safe commits
git config alias.safe-commit '!f() {
    echo "🧹 Running pre-commit checks...";
    cd packages && source venv/bin/activate && ruff check src/ tests/ --fix && ruff format src/ tests/ && pre-commit run --all-files &&
    cd .. && git add . && git commit -m "$1";
}; f'

# Add alias for just linting (no commit)
git config alias.lint '!f() {
    cd packages && source venv/bin/activate && ruff check src/ tests/ --fix && ruff format src/ tests/ && pre-commit run --all-files;
}; f'

# Add alias for quick fix
git config alias.fix '!f() {
    cd packages && source venv/bin/activate && ruff check src/ tests/ --fix && ruff format src/ tests/;
}; f'

echo ""
echo "✅ Git aliases created! You can now use:"
echo ""
echo "   git fix              - Quick auto-fix with Ruff"
echo "   git lint             - Full pre-commit check (no commit)"
echo "   git safe-commit 'msg' - Lint + auto-fix + commit in one command"
echo ""
echo "Example:"
echo "   git safe-commit 'Add new feature'"
echo ""

chmod +x setup-git-aliases.sh
