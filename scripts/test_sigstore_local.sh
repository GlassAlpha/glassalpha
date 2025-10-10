#!/usr/bin/env bash
# Test sigstore signing process locally to catch issues before CI
set -euo pipefail

echo "🔐 Testing sigstore signing process locally..."

# Check if cosign is available
if ! command -v cosign >/dev/null 2>&1; then
    echo "⚠️  cosign not installed locally - install it to test signing:"
    echo "   curl -O -L \"https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64\""
    echo "   sudo mv cosign-linux-amd64 /usr/local/bin/cosign"
    echo "   sudo chmod +x /usr/local/bin/cosign"
    echo ""
    echo "Skipping sigstore tests (cosign not available)"
    exit 0
fi

echo "✅ cosign found: $(cosign version)"

# Create test artifacts
TEST_DIR="/tmp/glassalpha-sigstore-test"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "📦 Creating test artifacts..."

# Create a test wheel (dummy file)
TEST_WHEEL="$TEST_DIR/test-1.0.0-py3-none-any.whl"
echo "Dummy wheel content" > "$TEST_WHEEL"

# Create a test SBOM (dummy JSON)
TEST_SBOM="$TEST_DIR/sbom.json"
cat > "$TEST_SBOM" << 'EOF'
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:test",
  "version": 1,
  "components": [
    {
      "type": "library",
      "name": "test-package",
      "version": "1.0.0"
    }
  ]
}
EOF

echo "✅ Created test artifacts:"
ls -la "$TEST_DIR/"

# Test signing process
echo ""
echo "🔐 Testing artifact signing..."

for file in "$TEST_WHEEL" "$TEST_SBOM"; do
    echo "Signing $(basename "$file")..."
    if cosign sign-blob --yes "$file" --output-signature "${file}.sig" >/dev/null 2>&1; then
        echo "   ✅ $(basename "$file") signed successfully"
        echo "   📄 Signature: $(basename "${file}.sig")"
        ls -lh "${file}.sig"
    else
        echo "   ❌ Failed to sign $(basename "$file")"
        exit 1
    fi
done

# Test signature verification
echo ""
echo "🔍 Testing signature verification..."

for file in "$TEST_WHEEL" "$TEST_SBOM"; do
    sig_file="${file}.sig"
    echo "Verifying $(basename "$file")..."
    if cosign verify-blob --signature "$sig_file" "$file" >/dev/null 2>&1; then
        echo "   ✅ $(basename "$file") signature verified"
    else
        echo "   ❌ Failed to verify $(basename "$file")"
        exit 1
    fi
done

# Clean up
rm -rf "$TEST_DIR"

echo ""
echo "✅ All sigstore tests passed!"
echo ""
echo "The CI signing process should work correctly."
echo "If this script fails, check:"
echo "  1. cosign installation and version"
echo "  2. Network connectivity for keyless signing"
echo "  3. File permissions"
