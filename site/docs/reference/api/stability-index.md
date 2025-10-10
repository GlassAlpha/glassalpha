# API Stability Index

This document defines the stability levels for GlassAlpha APIs and the breaking change policy.

---

## Overview

GlassAlpha uses a **three-level stability index** inspired by Node.js and Rust:

- **Stable**: Guaranteed backwards compatibility
- **Beta**: Feature-complete but may change before v1.0
- **Experimental**: Exploratory, expect changes

---

## Stability Levels

### 🟢 Stable (v0.2+)

**Definition**: Public APIs that will not change in backwards-incompatible ways without a major version bump.

**Breaking change policy**:

- Requires major version bump (e.g., 0.x → 1.0, 1.x → 2.0)
- Deprecation warnings for at least one minor version
- Migration guide provided

**Stable APIs**:

| API                           | Stability | Since | Notes                             |
| ----------------------------- | --------- | ----- | --------------------------------- |
| `ga.audit.from_model()`       | 🟢 Stable | v0.2  | Signature locked                  |
| `ga.audit.from_predictions()` | 🟢 Stable | v0.2  | Signature locked                  |
| `ga.audit.from_config()`      | 🟢 Stable | v0.2  | Signature locked                  |
| `AuditResult` attributes      | 🟢 Stable | v0.2  | `.performance`, `.fairness`, etc. |
| `AuditResult.to_pdf()`        | 🟢 Stable | v0.2  | Signature locked                  |
| `AuditResult.to_json()`       | 🟢 Stable | v0.2  | Signature locked                  |
| `AuditResult.save()`          | 🟢 Stable | v0.2  | Signature locked                  |
| Error codes (GAE\*)           | 🟢 Stable | v0.2  | Codes never reused                |

**Guarantees**:

- Parameter names and types locked
- Return types locked
- Error codes never reused
- Result JSON schema backwards-compatible
- Manifest schema backwards-compatible

---

### 🟡 Beta (v0.2)

**Definition**: Feature-complete and well-tested, but may change before v1.0 based on user feedback.

**Breaking change policy**:

- May change in minor versions (e.g., 0.2 → 0.3)
- Changelog will document all changes
- Migration guide provided if change is complex

**Beta APIs**:

| API                            | Stability | Since | Expected Stable                 |
| ------------------------------ | --------- | ----- | ------------------------------- |
| `AuditResult.equals()`         | 🟡 Beta   | v0.2  | v0.3 (after tolerance feedback) |
| `AuditResult.summary()`        | 🟡 Beta   | v0.2  | v0.3 (after format feedback)    |
| `ReadonlyMetrics` plot methods | 🟡 Beta   | v0.2  | v0.4 (plot API stabilization)   |
| Metric tolerance policy        | 🟡 Beta   | v0.2  | v0.3 (after validation)         |
| Data hashing algorithm         | 🟡 Beta   | v0.2  | v1.0 (critical for audits)      |

**Why Beta**:

- Tolerance values may need tuning based on real-world usage
- Plot API may evolve (switch to Plotly, add interactivity)
- Data hashing algorithm needs cross-platform validation

---

### 🔴 Experimental (v0.2)

**Definition**: Exploratory features that may change significantly or be removed.

**Breaking change policy**:

- May change or be removed in any version
- No migration guide required
- Use at your own risk

**Experimental APIs**:

| API                         | Stability       | Since | Notes                               |
| --------------------------- | --------------- | ----- | ----------------------------------- |
| `AuditResult._repr_html_()` | 🔴 Experimental | v0.2  | Notebook display format may change  |
| `AuditResult.__hash__()`    | 🔴 Experimental | v0.2  | Hashing implementation may change   |
| Internal canonicalization   | 🔴 Experimental | v0.2  | Algorithm may change for edge cases |

**Why Experimental**:

- `_repr_html_()`: Display format is aesthetic, not contractual
- `__hash__()`: Python's hash stability guarantees are limited
- Canonicalization: May need adjustments for new dtypes

---

## Breaking Change Policy

### What Counts as Breaking?

**Breaking changes (require major version bump)**:

- Removing a parameter
- Changing a parameter type
- Changing return type
- Renaming an error code
- Changing error message structure
- Changing JSON schema (non-additive)
- Changing manifest schema (non-additive)
- Changing data hashing algorithm

**Non-breaking changes (allowed in minor versions)**:

- Adding new optional parameters (with defaults)
- Adding new return fields (JSON/manifest)
- Adding new error codes
- Improving error messages (keeping structure)
- Adding new metrics
- Deprecating features (with warnings)

### Deprecation Process

When deprecating a Stable API:

1. **Deprecation warning added** (minor version N)

   - Feature still works
   - Warning emitted on use
   - Docs updated with migration guide

2. **Deprecation maintained** (minor version N+1)

   - Feature still works
   - Warning still emitted

3. **Removal** (major version N+1 → M)
   - Feature removed
   - Clear migration guide in CHANGELOG

**Example**:

```python
# v0.3: Deprecation warning
warnings.warn(
    "parameter 'old_param' is deprecated and will be removed in v1.0. "
    "Use 'new_param' instead. See: https://glassalpha.com/migration/old-param",
    DeprecationWarning,
    stacklevel=2
)

# v1.0: Removal
# Parameter no longer exists
```

---

## Version Numbering

GlassAlpha uses **Semantic Versioning** (semver):

- **Major** (X.0.0): Breaking changes to Stable APIs
- **Minor** (0.X.0): New features, breaking changes to Beta/Experimental
- **Patch** (0.0.X): Bug fixes, no API changes

**Pre-1.0 Exception**:

- Before v1.0, minor versions (0.X) may contain breaking changes
- Patch versions (0.X.Y) are always backwards-compatible
- All Stable APIs in v0.2+ have a migration guide if changed

---

## Upgrade Recommendations

### For Production Use

**Use only Stable APIs**:

```python
# ✅ Safe for production (Stable)
import glassalpha as ga

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender},
    random_seed=42
)

# ✅ Safe for production (Stable)
result.to_pdf("audit.pdf")
result.save("result.json")
```

**Avoid Beta/Experimental in production**:

```python
# ⚠️ Beta: May change (use with caution)
result.equals(other_result, rtol=1e-5)

# ❌ Experimental: May change significantly
html = result._repr_html_()
```

### For Development/Exploration

**Beta APIs are safe for notebooks**:

```python
# 🟡 Beta: Great for exploration, may change later
result.summary()
result.performance.plot()
```

**Experimental APIs**: Use but don't rely on

```python
# 🔴 Experimental: Handy for debugging, don't depend on format
display(result)  # Uses _repr_html_()
```

---

## Stability Lifecycle

### Path to Stability

```
Experimental → Beta → Stable
   ↓             ↓
Removed    Deprecated → Removed
```

**Typical timeline**:

- **Experimental**: 1-2 releases (0.2, 0.3)
- **Beta**: 2-3 releases (0.3, 0.4, 0.5)
- **Stable**: v1.0+

**Early graduation**: Features may become Stable before v1.0 if:

- High confidence in design
- Extensive real-world testing
- Critical for compliance (e.g., data hashing)

---

## Communicating Changes

### Changelog Format

```markdown
## v0.3.0

### Breaking Changes (Beta APIs)

- **[BREAKING]** `AuditResult.equals()`: Changed default `rtol` from 1e-5 to 1e-4
  - **Migration**: Explicitly pass `rtol=1e-5` for old behavior
  - **Reason**: Cross-platform floating-point differences
  - **Docs**: [Tolerance Policy](https://glassalpha.com/reference/tolerance)

### Deprecations (Stable APIs)

- **[DEPRECATED]** `from_model(old_param=...)` deprecated, use `new_param`
  - **Removal**: v1.0
  - **Migration**: [Guide](https://glassalpha.com/migration/old-param)

### New Features

- Added `AuditResult.to_csv()` export (Beta)

### Bug Fixes

- Fixed NaN handling in categorical protected attributes
```

---

## API Stability Checklist

Before marking an API as **Stable**:

- [ ] Feature-complete (no known missing functionality)
- [ ] Well-tested (contract tests + edge cases)
- [ ] Documented (API reference + examples)
- [ ] Real-world usage (3+ users in production)
- [ ] Cross-platform validated (Linux + macOS + Windows)
- [ ] Performance acceptable (no known bottlenecks)
- [ ] Type-safe (mypy --strict passes)
- [ ] Error handling complete (all error codes defined)

Before marking an API as **Beta**:

- [ ] Feature-complete
- [ ] Well-tested
- [ ] Documented
- [ ] No known major bugs

**Experimental** has no checklist (exploratory only).

---

## Questions?

If you're unsure about an API's stability:

1. **Check this document** for explicit stability levels
2. **Check docstrings** for `.. versionadded::` and `.. stability::` tags
3. **Ask on GitHub Discussions**: "Is X stable enough for production?"

**General rule**: If it's not listed as Stable above, assume Beta or Experimental.

---

## Related Documentation

- [API Reference](index.md) - Full API documentation
- [Changelog](https://github.com/GlassAlpha/glassalpha/blob/main/CHANGELOG.md) - Version history
- [Tolerance Policy](tolerance-policy.md) - Default tolerances for metric comparisons

---

**Last Updated**: 2025-10-07
**Applies to**: v0.2.0+
