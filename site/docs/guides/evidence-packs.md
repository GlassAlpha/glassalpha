# Evidence Pack Export & Verification

Complete guide to creating and verifying tamper-evident evidence packs for regulatory submissions.

---

## Overview

Evidence packs are tamper-evident ZIP bundles containing all audit artifacts, checksums, and verification instructions. They enable:

- **Regulatory submissions**: Package complete audit documentation for regulators
- **Independent verification**: Auditors can verify integrity without GlassAlpha installed
- **Chain of custody**: Cryptographic checksums prove authenticity
- **Reproducibility**: All artifacts needed to reproduce audits

## Quick start

```bash
# 1. Run audit
glassalpha audit --config audit.yaml --output audit.html

# 2. Export evidence pack
glassalpha export-evidence-pack audit.html --config audit.yaml

# 3. Verify integrity
glassalpha verify-evidence-pack audit_evidence_pack.zip
```

---

## What's included in an evidence pack

Every evidence pack contains:

### Core artifacts

- **Audit report**: HTML or PDF with complete analysis
- **Manifest**: Provenance tracking (hashes, versions, seeds)
- **Configuration**: YAML config used to generate audit
- **SHA256SUMS.txt**: Cryptographic checksums for all files
- **VERIFY.txt**: Instructions for independent verification

### Optional artifacts

- **Policy decision**: PASS/FAIL results for compliance gates (planned for v0.3.0)
- **Shift analysis**: Demographic shift testing results (if `--check-shift` used)
- **Badge**: SVG badge with audit status for sharing

### Metadata

- **canonical.jsonl**: Structured manifest of all pack contents
- **timestamps**: ISO 8601 UTC timestamps for all artifacts
- **tool_version**: GlassAlpha version used

---

## Creating evidence packs

### Basic usage

```bash
# Auto-generate pack name based on report
glassalpha export-evidence-pack audit.html
# Creates: audit_evidence_pack.zip
```

### Custom output path

```bash
# Specify output location
glassalpha export-evidence-pack audit.html --output compliance/q4_2024.zip
```

### Include configuration

```bash
# Bundle original config for reproducibility
glassalpha export-evidence-pack audit.html --config audit_config.yaml
```

### Skip badge generation

```bash
# Faster export without badge (for internal use)
glassalpha export-evidence-pack audit.html --no-badge
```

---

## Verification procedures

### Automated verification

```bash
# Verify all checksums and structure
glassalpha verify-evidence-pack evidence_pack.zip

# Verbose output with detailed log
glassalpha verify-evidence-pack pack.zip --verbose
```

**Exit codes:**

- `0`: Verification successful (all checksums match)
- `1`: Verification failed (tampering detected or corrupted)

### CI/CD integration

```bash
# Fail build if verification fails
glassalpha verify-evidence-pack evidence.zip || exit 1
```

### Manual verification

For auditors without GlassAlpha installed:

```bash
# 1. Extract pack
unzip audit_evidence_pack.zip -d audit_review/
cd audit_review/

# 2. Verify checksums (Linux/macOS)
sha256sum -c SHA256SUMS.txt

# 3. Verify checksums (Windows PowerShell)
Get-Content SHA256SUMS.txt | ForEach-Object {
    $hash, $file = $_ -split '\s+', 2
    $computed = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($hash -eq $computed) {
        Write-Host "OK: $file"
    } else {
        Write-Host "FAIL: $file"
    }
}
```

---

## Use cases by role

### Compliance officers

**Regulatory submission:**

```bash
# 1. Generate audit with all required features
glassalpha audit --config production_audit.yaml --output q4_audit.html --strict

# 2. Create evidence pack with config
glassalpha export-evidence-pack q4_audit.html --config production_audit.yaml \
  --output submissions/banking_q4_2024.zip

# 3. Verify before submission
glassalpha verify-evidence-pack submissions/banking_q4_2024.zip
```

**Response to regulator inquiry:**

```bash
# Package specific audit requested by regulator
glassalpha export-evidence-pack reports/credit_model_audit.html \
  --config configs/credit_model.yaml \
  --output regulator_response/model_validation_2024.zip
```

### Model validators

**Independent verification workflow:**

```bash
# 1. Receive evidence pack from organization
# 2. Verify integrity
glassalpha verify-evidence-pack submitted_pack.zip --verbose

# 3. Extract and review
unzip submitted_pack.zip -d review/
open review/audit.html  # Review findings

# 4. Reproduce audit (if config and data provided)
glassalpha audit --config review/audit_config.yaml --output reproduced.html

# 5. Compare hashes
diff <(sha256sum review/audit.html) <(sha256sum reproduced.html)
```

### ML engineers

**CI/CD audit archiving:**

```bash
# Save evidence pack as CI artifact
glassalpha audit --config ci.yaml --output audit.html
glassalpha export-evidence-pack audit.html --config ci.yaml \
  --output artifacts/audit_${CI_COMMIT_SHA}.zip
```

**Version comparison:**

```bash
# Create packs for model v1 and v2
glassalpha export-evidence-pack v1_audit.html --output v1_pack.zip
glassalpha export-evidence-pack v2_audit.html --output v2_pack.zip

# Compare manifests
unzip -p v1_pack.zip audit.manifest.json | jq .
unzip -p v2_pack.zip audit.manifest.json | jq .
```

---

## Industry-specific workflows

### Banking (SR 11-7 compliance)

```bash
# Quarterly model validation
glassalpha audit --config credit_model.yaml --output q4_audit.html --strict

# Create evidence pack for MRM
glassalpha export-evidence-pack q4_audit.html --config credit_model.yaml \
  --output model_risk_management/credit_q4_2024.zip

# Verification by independent validator
glassalpha verify-evidence-pack model_risk_management/credit_q4_2024.zip
```

**What to submit:**

- Evidence pack ZIP
- Cover letter citing specific SR 11-7 sections
- Model change documentation (if applicable)

[See SR 11-7 mapping guide →](../compliance/sr-11-7-mapping.md)

### Insurance (NAIC Model Act)

```bash
# Annual rate filing
glassalpha audit --config pricing_model.yaml --output rate_audit.html

# Package for state DOI submission
glassalpha export-evidence-pack rate_audit.html --config pricing_model.yaml \
  --output doi_filing/pricing_model_2024.zip
```

**State-specific notes:**

- CA Prop 103: Include proxy analysis and rate parity
- NY Reg 187: Document suitability model fairness
- CO SB 21-169: Include external data bias analysis

[See insurance guide →](../compliance/insurance-guide.md)

### Healthcare (HIPAA + health equity)

```bash
# Clinical validation
glassalpha audit --config risk_model.yaml --output clinical_audit.html

# Create pack for IRB or quality committee
glassalpha export-evidence-pack clinical_audit.html --config risk_model.yaml \
  --output irb_submission/risk_stratification_2024.zip
```

**Include with submission:**

- Clinical validation protocol
- Health equity analysis
- Disparate impact documentation

[See healthcare guide →](../compliance/healthcare-guide.md)

### Fraud detection (FCRA)

```bash
# Adverse action audit
glassalpha audit --config fraud_model.yaml --output fraud_audit.html

# Package with reason code validation
glassalpha export-evidence-pack fraud_audit.html --config fraud_model.yaml \
  --output compliance/adverse_action_2024.zip
```

[See fraud detection guide →](../compliance/fraud-guide.md)

---

## Troubleshooting

### Missing manifest file

**Error:**

```
Error: Manifest file not found: audit.manifest.json
```

**Fix:**

Manifests are auto-generated during audit. Re-run audit:

```bash
glassalpha audit --config audit.yaml --output audit.html
```

### Pack size too large

**Issue:** Evidence pack >100MB

**Solutions:**

1. **Skip badge** (saves ~50KB):

   ```bash
   glassalpha export-evidence-pack audit.html --no-badge
   ```

2. **Use HTML instead of PDF** (PDFs with plots can be 10-50MB):

   ```bash
   glassalpha audit --config audit.yaml --output audit.html --fast
   ```

3. **Compress separately** (for very large audits):
   ```bash
   # Extract without compression, then compress with higher ratio
   glassalpha export-evidence-pack audit.html --output pack.zip
   ```

### Verification fails

**Error:**

```
Verification failed: 1 file(s) with mismatched checksums
```

**Causes:**

- File was modified after pack creation
- Pack was corrupted during transfer
- Incomplete download

**Fix:**

Re-download or re-create pack. Verification failures indicate tampering or corruption.

### Windows path issues

**Issue:** Backslashes in paths cause verification failures

**Fix:**

Use forward slashes even on Windows:

```bash
glassalpha export-evidence-pack reports/audit.html --output compliance/pack.zip
```

---

## Advanced usage

### Batch processing

```bash
# Create packs for all audits in directory
for audit in reports/*.html; do
    glassalpha export-evidence-pack "$audit" \
        --output "evidence_packs/$(basename $audit .html)_pack.zip"
done
```

### Automated CI workflow

```yaml
# .github/workflows/audit.yml
- name: Generate audit
  run: glassalpha audit --config ci.yaml --output audit.html

- name: Create evidence pack
  run: glassalpha export-evidence-pack audit.html --output pack.zip

- name: Verify integrity
  run: glassalpha verify-evidence-pack pack.zip

- name: Upload artifact
  uses: actions/upload-artifact@v3
  with:
    name: audit-evidence-pack
    path: pack.zip
    retention-days: 90
```

### Timestamping (external)

For regulatory submissions requiring third-party timestamps:

```bash
# 1. Create pack
glassalpha export-evidence-pack audit.html --output pack.zip

# 2. Timestamp with RFC 3161 TSA (example)
openssl ts -query -data pack.zip -cert -sha256 -out pack.tsq
curl -H "Content-Type: application/timestamp-query" \
     --data-binary @pack.tsq \
     https://timestamp.example.com/ -o pack.tsr

# 3. Verify timestamp
openssl ts -verify -data pack.zip -in pack.tsr -CAfile ca.pem
```

### Custom verification scripts

```python
# verify_pack.py - Custom verification logic
import zipfile
import hashlib
import json

def verify_evidence_pack(pack_path):
    """Verify evidence pack with custom checks."""
    with zipfile.ZipFile(pack_path, 'r') as zf:
        # Check required files
        required = ['audit.html', 'audit.manifest.json', 'SHA256SUMS.txt']
        for file in required:
            if file not in zf.namelist():
                raise ValueError(f"Missing required file: {file}")

        # Verify checksums
        checksums = zf.read('SHA256SUMS.txt').decode().splitlines()
        for line in checksums:
            expected_hash, filename = line.split(None, 1)
            if filename in zf.namelist():
                data = zf.read(filename)
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(f"Checksum mismatch: {filename}")

        # Verify manifest structure
        manifest = json.loads(zf.read('audit.manifest.json'))
        assert 'tool_version' in manifest
        assert 'environment' in manifest

    print("✓ Verification successful")

if __name__ == "__main__":
    verify_evidence_pack("audit_evidence_pack.zip")
```

---

## Security considerations

### Chain of custody

Evidence packs include checksums but not digital signatures by default. For regulatory submissions requiring signatures:

1. **External signing** (recommended):

   ```bash
   # Create pack
   glassalpha export-evidence-pack audit.html --output pack.zip

   # Sign with GPG
   gpg --detach-sign --armor pack.zip
   # Creates: pack.zip.asc

   # Verify signature
   gpg --verify pack.zip.asc pack.zip
   ```

2. **KMS integration** (enterprise feature, planned):
   - AWS KMS signing
   - Azure Key Vault integration
   - HSM support

### Tamper detection

Verification checks detect:

- Modified files (checksum mismatch)
- Added files (not in SHA256SUMS.txt)
- Removed files (referenced but missing)
- Corrupted ZIP structure

**Limitations:**

- Does not prevent deletion of entire pack
- Does not prevent replay attacks (old pack presented as new)

**Mitigations:**

- Store packs in append-only storage
- Include timestamps from trusted TSA
- Implement version control for audit history

---

## Best practices

### Before submission

- [ ] Run audit in strict mode (`--strict`)
- [ ] Verify evidence pack (`verify-evidence-pack`)
- [ ] Include original configuration file
- [ ] Test extraction on clean system
- [ ] Document any warnings or limitations

### Naming conventions

```bash
# Recommended format: {model}_{purpose}_{date}.zip
credit_model_q4_validation_2024_10_14.zip
fraud_model_regulatory_submission_2024_10.zip
pricing_model_rate_filing_2024.zip
```

### Storage recommendations

- **Retention**: 5-7 years (matches regulatory audit periods)
- **Location**: Immutable storage (S3 with object lock, Azure immutable blobs)
- **Backup**: Keep offsite copies for disaster recovery
- **Access control**: Restrict to compliance team only

### Documentation

Include these with every submission:

1. **Cover letter**: Cite specific regulatory requirements
2. **Change log**: Document model changes since last audit
3. **Known limitations**: Be upfront about edge cases
4. **Verification instructions**: Include VERIFY.txt contents

---

## Related guides

- [CLI Commands](../reference/cli.md) - Complete CLI reference
- [Compliance Workflows](compliance-workflow.md) - Role-specific workflows
- [SR 11-7 Mapping](../compliance/sr-11-7-mapping.md) - Banking compliance
- [Determinism](determinism.md) - Reproducibility requirements
- [Validator Workflow](validator-workflow.md) - Independent verification

---

## FAQ

### Can I verify a pack without installing GlassAlpha?

Yes. Evidence packs include `VERIFY.txt` with manual verification steps using standard tools (`sha256sum`, `unzip`). See [manual verification](#manual-verification).

### What if I lost the original config?

The config is included in the evidence pack if you used `--config` during export. Extract and reuse:

```bash
unzip -j audit_pack.zip audit_config.yaml
```

### How do I reproduce an audit from an evidence pack?

Extract the pack, then run:

```bash
glassalpha audit --config audit_config.yaml --output reproduced.html
```

Compare hashes to verify byte-identical reproduction.

### Can I add custom files to an evidence pack?

Not directly. Create your own ZIP with the evidence pack and additional files:

```bash
# Create evidence pack
glassalpha export-evidence-pack audit.html --output base_pack.zip

# Create custom bundle
mkdir bundle
unzip base_pack.zip -d bundle/
cp custom_docs/* bundle/
cd bundle && zip -r ../final_submission.zip *
```

### What happens if verification fails?

Exit code 1 indicates tampering or corruption. Do not submit failed packs. Regenerate the audit and create a new pack.

### Are evidence packs required for compliance?

Not universally. Check your regulatory requirements:

- **SR 11-7**: Evidence packs demonstrate "comprehensive validation documentation"
- **NAIC**: Facilitates "independent review"
- **HIPAA**: Supports audit trail requirements
- **FCRA**: Not explicitly required but demonstrates due diligence

### Can I password-protect evidence packs?

Not natively. Use OS-level encryption:

```bash
# macOS: Create encrypted DMG
hdiutil create -encryption -srcfolder audit_pack.zip audit_encrypted.dmg

# Linux: Use GPG
gpg -c audit_pack.zip  # Creates audit_pack.zip.gpg

# Windows: Use 7-Zip with password
7z a -p audit_encrypted.7z audit_pack.zip
```

---

## Version notes

**Available in:** v0.2.1+

**Planned enhancements (v0.3.0):**

- Policy decision logs (E1: Policy-as-Code Gates)
- Native signing integration (KMS/HSM)
- Pack comparison tools
- Batch export/verification commands
