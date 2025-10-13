# Pre-Commit Workflow

## Problem Solved

AI agents create temporary files (`temp-*`, `UX_*.md`, etc.) that break `check-manifest` in CI if accidentally committed. This was causing CI failures that should be caught locally.

## Solution

`check-manifest` now runs as a pre-commit hook, catching orphan files **before** they reach CI.

## What Gets Caught

The hook validates that every file is either:

- Explicitly included in `MANIFEST.in` (for distribution)
- Explicitly excluded in `MANIFEST.in` (development-only)

## Files Auto-Excluded

These patterns are in `.gitignore` and `MANIFEST.in`:

```
temp-*                  # AI-generated test projects/files
UX_*.md                 # AI-generated UX documentation
*_VALIDATION.md         # AI-generated validation docs
*_COMPLETE.md           # AI-generated completion docs
*_SUMMARY.md            # AI-generated summaries
*_IMPROVEMENTS*.md      # AI-generated improvement docs
CI_*.md                 # AI-generated CI analysis
*_ANALYSIS.md           # AI-generated analysis docs
*_PLAN.md               # AI-generated plans
fix-*.plan.md           # AI-generated fix plans
```

## Agent Instructions

When you create ANY temporary file in the project root:

1. **Always** use `temp-` prefix (e.g., `temp-test-project/`, `temp-audit.html`)
2. This ensures `.gitignore` catches it
3. Pre-commit hook will catch it if you try to commit

## Testing Pre-Commit

```bash
# This should be blocked by .gitignore
echo "test" > UX_TEST.md
git add UX_TEST.md  # ❌ Fails: file is gitignored

# This should be caught by pre-commit hook
echo "test" > ORPHAN_FILE.md
git add ORPHAN_FILE.md
git commit -m "test"  # ❌ Fails: check-manifest catches it
```

## Developer Workflow

Pre-commit runs automatically on every commit. If it fails:

```bash
# Hook output shows what's wrong:
check-manifest...........Failed
missing from sdist:
  ORPHAN_FILE.md

suggested MANIFEST.in rules:
  include *.md
```

**Fix options:**

1. Delete the orphan file if temporary: `rm ORPHAN_FILE.md`
2. Add exclusion to `MANIFEST.in` if it's a new pattern: `exclude ORPHAN_*.md`
3. Add inclusion if it should be distributed: `include ORPHAN_FILE.md`

## Why This Exists

**Security**: Prevents accidentally shipping credentials, temp files, or development artifacts in the published package.

**CI efficiency**: Catches issues locally (seconds) instead of CI (minutes).

**Cleanliness**: Forces explicit decisions about what goes in the distribution.

## Bypass (Emergency Only)

```bash
# Skip pre-commit hook (not recommended)
git commit --no-verify -m "emergency fix"
```

Only use for genuine emergencies. The hook exists for a reason.
