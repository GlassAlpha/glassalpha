#!/bin/bash
# ============================================================================
# GlassAlpha - Cleanup Temporary Files
# ============================================================================
# This script removes common temporary files that shouldn't be committed.
# Run before commits to ensure clean git status.
#
# Usage: ./scripts/cleanup-temp-files.sh [dry-run]
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to dry-run mode if no arguments
DRY_RUN="${1:-dry-run}"

echo -e "${YELLOW}=== GlassAlpha Temporary File Cleanup ===${NC}"

if [[ "$DRY_RUN" == "dry-run" ]]; then
    echo "Running in DRY-RUN mode (no files will be deleted)"
    echo "Run with 'execute' to actually delete files"
    DELETE_CMD="echo Would delete:"
else
    echo "Running in EXECUTE mode (files will be deleted)"
    DELETE_CMD="rm -rf"
fi

echo ""

# Find and remove temporary files
TEMP_FILES=(
    "temp-*"           # AI-generated temp files (any extension)
    "*.tmp"
    "*.temp"
    "models/saved_model.*"      # Generated model artifacts
    "models/test_model.*"       # Test model artifacts
    # Note: examples/models/*.csv and *.meta.json are permanent example files, not temp files
    "models/quickstart_model.pkl"  # Specific generated model mentioned in .gitignore
)

# AI-generated test outputs (configs, reports, manifests)
AI_TEST_OUTPUTS=(
    # Test configs (both underscore and hyphen variants)
    "adult_config.yaml"
    "custom_config.yaml"
    "incomplete_config.yaml"
    "model_config.yaml"
    "pdf_config.yaml"
    "strict_config.yaml"
    "test_config.yaml"
    "xgboost_config.yaml"
    "german_credit.yaml"
    "german_credit_test.yaml"
    "*_test.yaml"           # Any dataset with _test suffix
    "test.yaml"
    "test-*.yaml"           # Hyphenated test configs
    "test-*.yml"            # Hyphenated test configs (yml extension)

    # Audit outputs (ANY config used for audit runs)
    "audit_*.html"          # Audit HTML outputs
    "audit_*.manifest.json" # Audit manifests
    "audit_*.shift_analysis.json" # Shift analysis outputs
    "*_test.html"           # Dataset test outputs (german_credit_test.html, etc.)
    "*_test.manifest.json"  # Dataset test manifests
    "*_report.html"
    "*_report.manifest.json"

    # Test outputs (hyphenated and underscored)
    "test_*.html"
    "test_*.pdf"
    "test_*.pkl"
    "test-*.html"           # Hyphenated test outputs
    "test-*.pdf"            # Hyphenated test outputs
    "test.html"
    "test.pdf"
    "test.manifest.json"

    # Quickstart outputs
    "quickstart.html"
    "quickstart.pdf"
    "quickstart.manifest.json"

    # Temporary markdown docs
    "COMMIT_MESSAGE.txt"
    "LABELING_RECOMMENDATIONS.md"
    "FINAL_UX_SUMMARY.md"
    "UX_FIXES_COMPLETE.md"
    "UX_IMPROVEMENTS_IMPLEMENTED.md"
    "UX_REVIEW_VALIDATED.md"
    "*_COMPLETE.md"
    "*_SUMMARY.md"
    "*_IMPROVEMENTS*.md"
    "*_VALIDATION.md"
    "UX_*.md"
)

echo -e "${YELLOW}Removing temporary files and generated artifacts:${NC}"

# Process temp files
for pattern in "${TEMP_FILES[@]}"; do
    if [[ "$pattern" == models/* ]]; then
        # Handle models/ subdirectory patterns - extract the filename part
        filename="${pattern#models/}"
        files=$(find models -maxdepth 1 -name "$filename" 2>/dev/null || true)
    elif [[ "$pattern" == examples/models/* ]]; then
        # Handle examples/models/ subdirectory patterns - extract the filename part
        filename="${pattern#examples/models/}"
        files=$(find examples/models -maxdepth 1 -name "$filename" 2>/dev/null || true)
    else
        # Handle root directory patterns
        files=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null || true)
    fi

    if [[ -n "$files" ]]; then
        while IFS= read -r file; do
            if [[ -e "$file" ]]; then
                echo "  $DELETE_CMD $file"
                [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$file"
            fi
        done <<< "$files"
    fi
done

# Process AI test outputs
echo -e "${YELLOW}Removing AI-generated test outputs from root:${NC}"
found_any=false
for pattern in "${AI_TEST_OUTPUTS[@]}"; do
    files=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null || true)
    if [[ -n "$files" ]]; then
        found_any=true
        while IFS= read -r file; do
            if [[ -f "$file" ]]; then
                echo "  $DELETE_CMD $file"
                [[ "$DRY_RUN" != "dry-run" ]] && rm -f "$file"
            fi
        done <<< "$files"
    fi
done

if [[ "$found_any" == "false" ]]; then
    echo "  (none found)"
fi

echo ""

# Clean .pytest_cache if it exists
if [[ -d ".pytest_cache" ]]; then
    echo -e "${YELLOW}Cleaning pytest cache:${NC}"
    echo "  $DELETE_CMD .pytest_cache"
    [[ "$DRY_RUN" != "dry-run" ]] && rm -rf .pytest_cache
fi

# Clean __pycache__ directories in root (but preserve in src/)
echo -e "${YELLOW}Cleaning root-level __pycache__ directories:${NC}"
root_pycache=$(find . -maxdepth 1 -name "__pycache__" -type d 2>/dev/null || true)
if [[ -n "$root_pycache" ]]; then
    while IFS= read -r dir; do
        echo "  $DELETE_CMD $dir"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$dir"
    done <<< "$root_pycache"
else
    echo "  (none found)"
fi

# Clean test project directories
echo -e "${YELLOW}Cleaning test project directories:${NC}"
TEST_DIRS=("my-audit-project" "notebook-test" "test-audit-project" "test-audit-project-2")
found_dirs=false
for dir in "${TEST_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        found_dirs=true
        echo "  $DELETE_CMD $dir/"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$dir"
    fi
done

# Also catch any test-audit-project-* pattern
test_audit_dirs=$(find . -maxdepth 1 -type d -name "test-audit-project*" 2>/dev/null || true)
if [[ -n "$test_audit_dirs" ]]; then
    found_dirs=true
    while IFS= read -r dir; do
        echo "  $DELETE_CMD $dir/"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$dir"
    done <<< "$test_audit_dirs"
fi

if [[ "$found_dirs" == "false" ]]; then
    echo "  (none found)"
fi

echo ""

# Clean duplicate project directory (AI-generated during testing)
echo -e "${YELLOW}Cleaning duplicate project structures:${NC}"
DUPLICATE_DIRS=("glassalpha")
found_duplicates=false
for dir in "${DUPLICATE_DIRS[@]}"; do
    if [[ -d "$dir" && -d "$dir/.git" ]]; then
        found_duplicates=true
        echo "  $DELETE_CMD $dir/ (duplicate Git repository)"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$dir"
    fi
done
if [[ "$found_duplicates" == "false" ]]; then
    echo "  (none found)"
fi

echo ""

# Clean build artifacts and generated outputs
echo -e "${YELLOW}Cleaning build artifacts:${NC}"
BUILD_ARTIFACTS=("build" "site/site" "glassalpha/site/site")
found_builds=false
for artifact in "${BUILD_ARTIFACTS[@]}"; do
    if [[ -d "$artifact" ]]; then
        found_builds=true
        echo "  $DELETE_CMD $artifact/"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$artifact"
    fi
done
if [[ "$found_builds" == "false" ]]; then
    echo "  (none found)"
fi

echo ""

# Note: dist/ is intentionally NOT cleaned here - it should be cleaned by `make clean`
# to preserve wheel artifacts until user explicitly rebuilds

echo ""

# Clean virtual environment directories (not in .gitignore but should be cleaned)
echo -e "${YELLOW}Cleaning stray virtual environments:${NC}"
VENV_DIRS=("glassalpha-env" "venv-glassalpha" "*_venv")
found_venvs=false
for venv_pattern in "${VENV_DIRS[@]}"; do
    if [[ "$venv_pattern" == *"*"* ]]; then
        # Handle wildcard patterns
        venv_matches=$(find . -maxdepth 1 -type d -name "$venv_pattern" 2>/dev/null || true)
        if [[ -n "$venv_matches" ]]; then
            found_venvs=true
            while IFS= read -r venv; do
                echo "  $DELETE_CMD $venv/"
                [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$venv"
            done <<< "$venv_matches"
        fi
    elif [[ -d "$venv_pattern" ]]; then
        # Handle exact directory names
        found_venvs=true
        echo "  $DELETE_CMD $venv_pattern/"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$venv_pattern"
    fi
done
if [[ "$found_venvs" == "false" ]]; then
    echo "  (none found)"
fi

echo ""

# Check for any remaining temp files that might need attention (exclude already processed files)
remaining=$(find . -maxdepth 2 \( -name "temp-*" -o -name "*.tmp" -o -name "*.temp" \) ! -path "./models/*" 2>/dev/null | head -10 || true)
if [[ -n "$remaining" ]]; then
    echo -e "${RED}Warning: Found additional temporary files that may need manual cleanup:${NC}"
    echo "$remaining"
    echo ""
    echo "Consider adding these patterns to this script or .gitignore"
fi

echo -e "${GREEN}Cleanup complete!${NC}"

if [[ "$DRY_RUN" == "dry-run" ]]; then
    echo ""
    echo "To actually delete files, run:"
    echo "  ./scripts/cleanup-temp-files.sh execute"
fi
