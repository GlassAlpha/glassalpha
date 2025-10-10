#!/bin/bash
# Test Stability Check Script
# Runs critical tests multiple times to detect flakiness and non-determinism
#
# Usage:
#   ./scripts/test_stability_check.sh [runs] [test_pattern]
#
# Examples:
#   ./scripts/test_stability_check.sh 10  # Run all critical tests 10 times
#   ./scripts/test_stability_check.sh 5 determinism  # Run determinism tests 5 times
#
# Exit codes:
#   0 - All runs passed, tests are stable
#   1 - At least one run failed, tests are flaky
#   2 - Configuration error

set -euo pipefail

# Configuration
RUNS="${1:-5}"  # Default: 5 runs
TEST_PATTERN="${2:-critical|determinism|notebook_api}"  # Default: critical tests
TEMP_DIR=$(mktemp -d)
RESULTS_FILE="${TEMP_DIR}/results.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup on exit
cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

echo "==================================================================================="
echo "GlassAlpha Test Stability Check"
echo "==================================================================================="
echo "Configuration:"
echo "  - Runs: ${RUNS}"
echo "  - Test pattern: ${TEST_PATTERN}"
echo "  - Results directory: ${TEMP_DIR}"
echo "==================================================================================="
echo ""

# Collect tests matching pattern
echo "Collecting tests..."
TEST_FILES=$(find tests -name "test_*.py" -type f | grep -E "${TEST_PATTERN}" || true)

if [ -z "${TEST_FILES}" ]; then
    echo -e "${RED}ERROR: No tests found matching pattern '${TEST_PATTERN}'${NC}"
    exit 2
fi

echo "Found tests:"
echo "${TEST_FILES}" | sed 's/^/  - /'
echo ""

# Run stability tests
FAILED_RUNS=0
PASSED_RUNS=0

for ((i=1; i<=RUNS; i++)); do
    echo "==================================================================================="
    echo "Run $i of ${RUNS}"
    echo "==================================================================================="

    # Run pytest with deterministic environment
    if PYTHONHASHSEED=0 \
       TZ=UTC \
       MPLBACKEND=Agg \
       SOURCE_DATE_EPOCH=1577836800 \
       OMP_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 \
       python3 -m pytest -k "${TEST_PATTERN}" --tb=short -v \
       > "${TEMP_DIR}/run_${i}.log" 2>&1; then
        echo -e "${GREEN}✓ Run $i: PASSED${NC}"
        PASSED_RUNS=$((PASSED_RUNS + 1))
        echo "PASS" >> "${RESULTS_FILE}"
    else
        echo -e "${RED}✗ Run $i: FAILED${NC}"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "FAIL" >> "${RESULTS_FILE}"

        # Show last 20 lines of failure log
        echo ""
        echo "Failure details (last 20 lines):"
        tail -n 20 "${TEMP_DIR}/run_${i}.log" | sed 's/^/  /'
        echo ""
    fi
done

echo ""
echo "==================================================================================="
echo "Stability Report"
echo "==================================================================================="
echo "Total runs: ${RUNS}"
echo -e "Passed: ${GREEN}${PASSED_RUNS}${NC}"
echo -e "Failed: ${RED}${FAILED_RUNS}${NC}"

if [ ${FAILED_RUNS} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Tests are stable! All ${RUNS} runs passed.${NC}"
    echo ""
    echo "Logs saved to: ${TEMP_DIR}"
    exit 0
else
    FAILURE_RATE=$(awk "BEGIN {printf \"%.1f\", ${FAILED_RUNS}/${RUNS}*100}")
    echo ""
    echo -e "${RED}✗ Tests are flaky! Failure rate: ${FAILURE_RATE}%${NC}"
    echo ""
    echo "Failure logs saved to: ${TEMP_DIR}"
    echo ""
    echo "To investigate:"
    echo "  1. Check logs: ls ${TEMP_DIR}/run_*.log"
    echo "  2. Compare outputs: diff ${TEMP_DIR}/run_1.log ${TEMP_DIR}/run_2.log"
    echo "  3. Look for non-deterministic patterns (timestamps, random values, etc.)"
    echo ""
    exit 1
fi
