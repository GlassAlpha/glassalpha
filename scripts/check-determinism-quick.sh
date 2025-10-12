#!/bin/bash
# Quick determinism check - run before pushing
# Tests that audit generation produces identical hashes across 3 runs
set -e

echo "========================================"
echo "Quick Determinism Check (3 runs)"
echo "========================================"
echo ""

# Set deterministic environment (same as CI)
export SOURCE_DATE_EPOCH=1577836800
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GLASSALPHA_DETERMINISTIC=1
export TZ=UTC
export MPLBACKEND=Agg

echo "Environment:"
echo "  SOURCE_DATE_EPOCH: $SOURCE_DATE_EPOCH"
echo "  PYTHONHASHSEED: $PYTHONHASHSEED"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  TZ: $TZ"
echo "  MPLBACKEND: $MPLBACKEND"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -f /tmp/audit_*.html
}
trap cleanup EXIT

echo "Running 3 audit generations..."
for i in {1..3}; do
  echo "  Run $i/3..."
  glassalpha audit -c audit_config.yaml -o /tmp/audit_${i}.html > /dev/null 2>&1
  sha256sum /tmp/audit_${i}.html
done | awk '{print $1}' | sort | uniq -c

echo ""
echo "========================================"
echo "Results:"
echo "========================================"
echo ""
echo "Expected: '3 <hash>' (all 3 runs identical)"
echo "If you see >1 unique hash, determinism is broken"
echo ""

# Check if all hashes are identical
unique_count=$(for i in {1..3}; do sha256sum /tmp/audit_${i}.html; done | awk '{print $1}' | sort | uniq | wc -l)

if [ "$unique_count" -eq 1 ]; then
    echo "✅ SUCCESS: All runs produced identical output"
    exit 0
else
    echo "❌ FAILURE: Runs produced $unique_count different hashes"
    echo ""
    echo "Hashes:"
    for i in {1..3}; do
        echo "  Run $i: $(sha256sum /tmp/audit_${i}.html | awk '{print $1}')"
    done
    echo ""
    echo "This indicates non-deterministic behavior."
    echo "Check:"
    echo "  1. Config has 'reproducibility.strict: true'"
    echo "  2. All random seeds are set"
    echo "  3. Thread counts are controlled (OMP_NUM_THREADS=1)"
    exit 1
fi
