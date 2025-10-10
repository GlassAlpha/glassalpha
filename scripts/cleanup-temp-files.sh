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
    "temp-*.py"
    "temp-*.txt"
    "temp-*.md"
    "*.tmp"
    "*.temp"
)

echo -e "${YELLOW}Removing temporary files from root directory:${NC}"

for pattern in "${TEMP_FILES[@]}"; do
    files=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null || true)
    if [[ -n "$files" ]]; then
        echo "$files" | while read -r file; do
            if [[ -f "$file" ]]; then
                echo "  $DELETE_CMD $file"
                [[ "$DRY_RUN" != "dry-run" ]] && rm -f "$file"
            fi
        done
    fi
done

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
    echo "$root_pycache" | while read -r dir; do
        echo "  $DELETE_CMD $dir"
        [[ "$DRY_RUN" != "dry-run" ]] && rm -rf "$dir"
    done
fi

echo ""

# Check for any remaining temp files that might need attention
remaining=$(find . -maxdepth 2 -name "temp-*" -o -name "*.tmp" -o -name "*.temp" 2>/dev/null | head -10 || true)
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
