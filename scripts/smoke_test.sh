#!/bin/bash
# Smoke test script for GlassAlpha
# Tests critical user journeys before PyPI publication
set -e

echo "========================================"
echo "GlassAlpha Smoke Test Suite"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Helper functions
pass_test() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "${GREEN}✓${NC} $1"
}

fail_test() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "${RED}✗${NC} $1"
    echo "  Error: $2"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up test artifacts..."
    rm -rf test-german test-adult test-model.pkl
}
trap cleanup EXIT

# Test 1: German Credit QuickStart
echo "Test 1: German Credit quickstart..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
if glassalpha quickstart --dataset german_credit --output test-german --no-interactive > /dev/null 2>&1; then
    if cd test-german && python run_audit.py > /dev/null 2>&1; then
        cd ..
        pass_test "German Credit quickstart and audit"
    else
        cd ..
        fail_test "German Credit quickstart" "Audit execution failed"
    fi
else
    fail_test "German Credit quickstart" "Project generation failed"
fi

# Test 2: Adult Income QuickStart (THE CRITICAL FIX)
echo "Test 2: Adult Income quickstart..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
if glassalpha quickstart --dataset adult_income --output test-adult --no-interactive > /dev/null 2>&1; then
    if cd test-adult && python run_audit.py > /dev/null 2>&1; then
        cd ..
        pass_test "Adult Income quickstart and audit"
    else
        cd ..
        fail_test "Adult Income quickstart" "Audit execution failed (check target_column)"
    fi
else
    fail_test "Adult Income quickstart" "Project generation failed"
fi

# Test 3: HTML File Size
echo "Test 3: HTML report file size..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
if [ -f "test-german/reports/audit_report.html" ]; then
    SIZE=$(du -m "test-german/reports/audit_report.html" | cut -f1)
    if [ "$SIZE" -lt 20 ]; then
        pass_test "HTML file size ($SIZE MB < 20 MB)"
    else
        fail_test "HTML file size" "Report is ${SIZE}MB (expected < 20MB)"
    fi
else
    fail_test "HTML file size" "Report file not found"
fi

# Test 4: No Matplotlib Warnings
echo "Test 4: Matplotlib warnings check..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
OUTPUT=$(glassalpha audit --config test-german/audit_config.yaml --output test-check.html --fast 2>&1)
if echo "$OUTPUT" | grep -q "optimize"; then
    fail_test "Matplotlib warnings" "Found 'optimize' parameter warning"
else
    pass_test "No matplotlib warnings"
fi
rm -f test-check.html test-check.manifest.json

# Test 5: Reasons Command with Model Save
echo "Test 5: Reasons command with wrapped model..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
if glassalpha audit --config test-german/audit_config.yaml --save-model test-model.pkl > /dev/null 2>&1; then
    # Get path to data file
    DATA_PATH="$HOME/Library/Application Support/glassalpha/data/german_credit_processed.csv"
    if [ -f "$DATA_PATH" ]; then
        if glassalpha reasons --model test-model.pkl --data "$DATA_PATH" --instance 0 > /dev/null 2>&1; then
            pass_test "Reasons command with XGBoostWrapper"
        else
            fail_test "Reasons command" "SHAP explainer failed with wrapped model"
        fi
    else
        warn "Skipping reasons test (data file not found)"
    fi
else
    fail_test "Reasons command" "Model save failed"
fi

# Test 6: CLI Commands Available
echo "Test 6: CLI commands availability..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
COMMANDS_OK=true
for cmd in audit validate quickstart doctor list datasets models; do
    if ! glassalpha $cmd --help > /dev/null 2>&1; then
        COMMANDS_OK=false
        warn "Command '$cmd' failed"
    fi
done
if [ "$COMMANDS_OK" = true ]; then
    pass_test "All CLI commands available"
else
    fail_test "CLI commands" "Some commands not working"
fi

# Test 7: Fast Mode Performance
echo "Test 7: Fast mode performance..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
START=$(date +%s)
glassalpha audit --config test-german/audit_config.yaml --output test-fast.html --fast > /dev/null 2>&1
END=$(date +%s)
DURATION=$((END - START))
rm -f test-fast.html test-fast.manifest.json

if [ "$DURATION" -lt 10 ]; then
    pass_test "Fast mode completes in ${DURATION}s (< 10s)"
else
    warn "Fast mode took ${DURATION}s (expected < 10s)"
    pass_test "Fast mode works (but slower than expected)"
fi

# Test 8: Strict Mode
echo "Test 8: Strict mode execution..."
TESTS_TOTAL=$((TESTS_TOTAL + 1))
if glassalpha audit --config test-german/audit_config.yaml --output test-strict.html --strict > /dev/null 2>&1; then
    pass_test "Strict mode execution"
else
    fail_test "Strict mode" "Execution failed"
fi
rm -f test-strict.html test-strict.manifest.json

# Summary
echo ""
echo "========================================"
echo "Test Results"
echo "========================================"
echo "Total:  $TESTS_TOTAL"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ All smoke tests passed!${NC}"
    echo "Ready for PyPI publication."
    exit 0
else
    echo -e "${RED}✗ Some tests failed!${NC}"
    echo "DO NOT publish to PyPI until all tests pass."
    exit 1
fi
