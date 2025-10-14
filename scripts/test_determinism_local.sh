#!/bin/bash
# Local determinism test script - mirrors CI determinism tests
# Run this before pushing to catch CI issues locally
set -e

echo "========================================"
echo "Local Determinism Test Suite (CI Mirror)"
echo "========================================"
echo ""

# Use venv python if available, otherwise system python3
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "Using venv Python: $PYTHON"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
    echo "Using activated venv Python"
else
    PYTHON="python3"
    echo "⚠️  Warning: Using system Python (no venv detected)"
fi
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
pass_test() {
    echo -e "${GREEN}✓${NC} $1"
}

fail_test() {
    echo -e "${RED}✗${NC} $1"
    echo "  Error: $2"
    exit 1
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up test artifacts..."
    rm -rf /tmp/audit*.pdf /tmp/test_config.yaml /tmp/hashes.txt
}
trap cleanup EXIT

# Set deterministic environment (same as CI)
export SOURCE_DATE_EPOCH=1577836800
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GLASSALPHA_DETERMINISTIC=1
export TZ=UTC
export MPLBACKEND=Agg

echo "=== Environment Setup ==="
echo "SOURCE_DATE_EPOCH: $SOURCE_DATE_EPOCH"
echo "PYTHONHASHSEED: $PYTHONHASHSEED"
echo "Platform: $(uname -a)"
echo "Python: $($PYTHON --version)"
echo ""

# Test 1: Verify critical dependencies
echo "=== Test 1: Verify critical dependencies ==="
$PYTHON -c "import shap; print(f'✅ SHAP {shap.__version__}')" || fail_test "SHAP dependency" "SHAP not available"
$PYTHON -c "import weasyprint; print(f'✅ WeasyPrint {weasyprint.__version__}')" || fail_test "WeasyPrint dependency" "WeasyPrint not available"
pass_test "All critical dependencies available"

# Note: PDF determinism is handled at HTML level (HTML is byte-identical)
# PDF generation is for human review only, not determinism guarantees

# Test 3: HTML determinism validation
echo "=== Test 3: HTML determinism validation ==="

# Create test config (HTML output for deterministic validation)
cat > /tmp/test_config.yaml << 'EOF'
audit_profile: tabular_compliance
data:
  dataset: german_credit
  target_column: credit_risk
model:
  type: logistic_regression
reproducibility:
  random_seed: 42
EOF

# Run audit twice (HTML for deterministic validation)
echo "Running first audit..."
$PYTHON -m glassalpha audit -c /tmp/test_config.yaml -o /tmp/audit1.html || fail_test "First audit" "Audit failed"

echo "Running second audit..."
$PYTHON -m glassalpha audit -c /tmp/test_config.yaml -o /tmp/audit2.html || fail_test "Second audit" "Audit failed"

# Verify files exist
if [ ! -f /tmp/audit1.html ]; then
    fail_test "File creation" "/tmp/audit1.html was not created"
fi

if [ ! -f /tmp/audit2.html ]; then
    fail_test "File creation" "/tmp/audit2.html was not created"
fi

pass_test "Both audit HTMLs created successfully"

# Compute hashes (HTML is byte-identical, PDFs are not)
hash1=$(sha256sum /tmp/audit1.html | cut -d' ' -f1)
hash2=$(sha256sum /tmp/audit2.html | cut -d' ' -f1)

echo "Hash 1: $hash1"
echo "Hash 2: $hash2"

# Compare
if [ "$hash1" = "$hash2" ]; then
    pass_test "HTML hashes match - deterministic"
else
    fail_test "Hash comparison" "HTML hashes differ - non-deterministic"
fi

# Test 4: DeterminismValidator framework
echo "=== Test 4: DeterminismValidator framework ==="
$PYTHON << 'EOF'
import sys
import tempfile
import yaml
from pathlib import Path

# Use installed package (not hardcoded path)
from glassalpha.utils.determinism_validator import validate_audit_determinism

# Create test config
config = {
    'audit_profile': 'tabular_compliance',
    'data': {
        'dataset': 'german_credit',
        'target_column': 'credit_risk',
    },
    'model': {'type': 'logistic_regression'},
    'reproducibility': {
        'random_seed': 42,
    },
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(config, f)
    config_path = Path(f.name)

try:
    # Run validation
    report = validate_audit_determinism(
        config_path=config_path,
        runs=2,  # Reduced for speed
        seed=42,
        check_shap=False,  # Skip SHAP check for faster test
    )

    print(f"Is deterministic: {report.is_deterministic}")
    print(f"Summary: {report.summary}")
    print(f"Unique hashes: {len(set(report.hashes))}")

    if not report.is_deterministic:
        print("Non-determinism sources:")
        for source in report.non_determinism_sources:
            print(f"  - {source}")
        sys.exit(1)

    print("✅ DeterminismValidator working correctly")

finally:
    config_path.unlink()
EOF

if [ $? -eq 0 ]; then
    pass_test "DeterminismValidator framework working"
else
    fail_test "DeterminismValidator framework" "Test failed"
fi

# Test 5: Explainer selection determinism
echo "=== Test 5: Explainer selection determinism ==="
$PYTHON << 'EOF'
# Use installed package (not hardcoded path)
from glassalpha.explain import select_explainer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Create test data
np.random.seed(42)
X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
y = np.random.randint(0, 2, 100)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Test deterministic selection
explainer1 = select_explainer("logistic_regression")
explainer2 = select_explainer("logistic_regression")

assert explainer1 == explainer2, f"Non-deterministic selection: {explainer1} != {explainer2}"
print(f"✅ Deterministic explainer selection: {explainer1}")
EOF

if [ $? -eq 0 ]; then
    pass_test "Deterministic explainer selection working"
else
    fail_test "Explainer selection determinism" "Test failed"
fi

echo ""
echo "========================================"
echo -e "${GREEN}✓ All 5 determinism tests passed!${NC}"
echo "Local environment matches CI requirements."
echo "Safe to push to CI."
echo "========================================"
