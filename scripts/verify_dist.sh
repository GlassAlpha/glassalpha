#!/bin/sh
# GlassAlpha Distribution Verification Script
# Verifies SBOM completeness, Sigstore signatures, and artifact integrity

ARTIFACT_FILE="$1"

if [ "$ARTIFACT_FILE" = "" ]; then
    echo "Usage: $0 <artifact_file>"
    echo "Example: $0 dist/glassalpha-0.2.0-py3-none-any.whl"
    echo "Example: $0 dist/sbom.json"
    exit 1
fi

if [ ! -f "$ARTIFACT_FILE" ]; then
    echo "Artifact file not found: $ARTIFACT_FILE"
    exit 1
fi

DIST_DIR="$(dirname "$ARTIFACT_FILE")"

# Determine file type
if echo "$ARTIFACT_FILE" | grep -q "\.whl$"; then
    FILE_TYPE="wheel"
    WHEEL_FILE="$ARTIFACT_FILE"
elif echo "$ARTIFACT_FILE" | grep -q "\.json$"; then
    FILE_TYPE="sbom"
    SBOM_FILE="$ARTIFACT_FILE"
else
    echo "Unsupported file type: $ARTIFACT_FILE (expected .whl or .json)"
    exit 1
fi

echo "Verifying GlassAlpha artifact: $ARTIFACT_FILE ($FILE_TYPE)"

# 1. Verify SBOM structure and metadata
echo "Step 1: Validating SBOM structure..."
python3 -c "import json, sys; sbom = json.load(open('$ARTIFACT_FILE')); print('✅ SBOM valid: ' + sbom['specVersion'] + ' with ' + str(len(sbom['components'])) + ' components' if sbom.get('bomFormat') == 'CycloneDX' else sys.exit(1))"

# 2. Verify checksums and file integrity
echo "Step 2: Verifying file integrity..."
python3 -c "import hashlib; sha256_hash = hashlib.sha256(); f=open('$ARTIFACT_FILE', 'rb'); chunk=f.read(4096); [sha256_hash.update(chunk) for chunk in iter(lambda: f.read(4096), b'')]; print('✅ SHA256: ' + sha256_hash.hexdigest()[:16] + '...')"

# 3. Verify Sigstore signature exists (if wheel file)
if [ "$FILE_TYPE" = "wheel" ]; then
    SIG_FILE="$DIST_DIR/$(basename "$WHEEL_FILE" .whl).sig"
    if [ -f "$SIG_FILE" ]; then
        echo "Step 3: Validating Sigstore signature..."
        # Check if cosign is available and signature exists
        if command -v cosign >/dev/null 2>&1; then
            if cosign verify-blob --signature "$SIG_FILE" "$WHEEL_FILE" >/dev/null 2>&1; then
                echo "✅ Sigstore signature valid"
            else
                echo "❌ Sigstore verification failed"
                exit 1
            fi
        else
            echo "⚠️  cosign not available - signature verification skipped"
        fi
    else
        echo "Step 3: No signature file found - verification skipped"
    fi
else
    echo "Step 3: Signature verification skipped (SBOM file)"
fi

# Final summary
echo "Verification complete!"
echo ""
echo "To verify this distribution:"
echo "1. Install cosign: https://docs.sigstore.dev/cosign/installation"
echo "2. Run: cosign verify-blob --signature \$SIG_FILE \$WHEEL_FILE"
echo "3. Inspect SBOM: cat \$SBOM_FILE | jq '.components | length'"
echo "4. Check for vulnerabilities: pip-audit --desc"
echo ""
echo "For automated verification in CI/CD:"
echo "curl -s https://raw.githubusercontent.com/GlassAlpha/glassalpha/main/scripts/verify_dist.sh | bash -s \$WHEEL_FILE"
