#!/bin/bash
# GlassAlpha Determinism Test Script
# Tests that audit generation produces identical hashes across multiple runs
#
# Usage:
#   ./scripts/test_determinism.sh [mode] [runs]
#
# Modes:
#   quick     - 3 runs (default, fast)
#   full      - 10 runs (comprehensive, slower)
#   ci        - CI-style test (10 runs, strict validation)
#
# Examples:
#   ./scripts/test_determinism.sh quick     # 3 runs
#   ./scripts/test_determinism.sh full 10   # 10 runs
#   ./scripts/test_determinism.sh ci        # CI mode
#
# Exit codes:
#   0 - All runs identical (determinism confirmed)
#   1 - Non-deterministic behavior detected
#   2 - Configuration or setup error

set -euo pipefail

# Configuration
MODE="${1:-quick}"
RUNS="${2:-}"
SCRIPT_DIR="$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set default runs based on mode
case "$MODE" in
    "quick")
        RUNS="${RUNS:-3}"
        ;;
    "full")
        RUNS="${RUNS:-10}"
        ;;
    "ci")
        RUNS="${RUNS:-10}"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Valid modes: quick, full, ci"
        exit 2
        ;;
esac

echo "🔬 GlassAlpha Determinism Test: $MODE mode"
echo "========================================"
echo "Running $RUNS audit generations..."
echo ""

# Source unified determinism environment
source "$SCRIPT_DIR/setup-determinism-env.sh"

# Verify environment setup
if [ -z "$GLASSALPHA_STRICT" ]; then
    echo "⚠️  Warning: GLASSALPHA_STRICT not set"
    echo "   Setting strict mode for testing..."
    export GLASSALPHA_STRICT=1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -f /tmp/audit_*.html /tmp/audit_*.json
}
trap cleanup EXIT

# Run audit generations
echo "Generating $RUNS audit reports..."
hashes=()

for i in $(seq 1 "$RUNS"); do
    echo "  Run $i/$RUNS..."
    output_file="/tmp/audit_${i}.html"

    # Run audit (capture stderr to avoid noise)
    if ! glassalpha audit -c audit_config.yaml -o "$output_file" >/dev/null 2>&1; then
        echo "❌ Audit generation failed on run $i"
        exit 1
    fi

    # Get hash
    hash=$(sha256sum "$output_file" | awk '{print $1}')
    hashes+=("$hash")

    if [ "$MODE" = "ci" ]; then
        echo "    Hash: ${hash:0:16}..."
    fi
done

# Analyze results
unique_hashes=$(printf '%s\n' "${hashes[@]}" | sort | uniq | wc -l)

echo ""
echo "========================================"
echo "Determinism Test Results ($MODE mode)"
echo "========================================"
echo ""

if [ "$unique_hashes" -eq 1 ]; then
    echo "✅ SUCCESS: All $RUNS runs produced identical output"
    echo "   Hash: ${hashes[0]:0:16}..."
    echo ""
    echo "✅ Determinism confirmed - audit outputs are byte-identical"
    exit 0
else
    echo "❌ FAILURE: $RUNS runs produced $unique_hashes different hashes"
    echo ""
    echo "Non-deterministic behavior detected!"
    echo ""

    if [ "$RUNS" -le 5 ]; then
        echo "Hash comparison:"
        for i in $(seq 1 "$RUNS"); do
            echo "  Run $i: ${hashes[$((i-1))]:0:16}..."
        done
    else
        echo "First 5 hashes:"
        for i in $(seq 1 5); do
            echo "  Run $i: ${hashes[$((i-1))]:0:16}..."
        done
        echo "  ... and $((RUNS-5)) more"
    fi

    echo ""
    echo "Troubleshooting:"
    echo "  1. Check config has 'reproducibility.strict: true'"
    echo "  2. Verify all random seeds are set (42)"
    echo "  3. Check thread counts (OMP_NUM_THREADS=1)"
    echo "  4. Ensure SOURCE_DATE_EPOCH is set"
    echo ""
    echo "Run with debug: GLASSALPHA_DEBUG=1 glassalpha audit -c audit_config.yaml -o /tmp/debug.html"

    exit 1
fi
