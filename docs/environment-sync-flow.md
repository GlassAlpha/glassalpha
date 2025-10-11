# Environment Synchronization Flow

## Visual Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     You run: make test                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  check-venv runs      │
                │  automatically        │
                └───────┬───────────────┘
                        │
                        ▼
            ┌──────────────────────────┐
            │ Is venv in sync?         │
            │ (editable mode?)         │
            └────┬──────────────┬──────┘
                 │              │
            YES  │              │  NO
                 │              │
                 ▼              ▼
         ┌───────────┐   ┌────────────────────────┐
         │ ✅ Pass   │   │ ❌ FAIL with message:  │
         │ Run tests │   │                        │
         └───────────┘   │ "Environment out of    │
                         │  sync - package not    │
                         │  in editable mode"     │
                         │                        │
                         │ Options shown:         │
                         │ 1. make sync-deps      │
                         │ 2. make test AUTO_FIX=1│
                         │ 3. pip install -e .    │
                         └────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
               Option 1 │              Option 2 │
                        ▼                       ▼
            ┌─────────────────┐     ┌──────────────────┐
            │ make sync-deps  │     │ make test        │
            │ (full reset)    │     │ AUTO_FIX=1       │
            └────────┬────────┘     │ (auto-fix)       │
                     │              └────────┬─────────┘
                     │                       │
                     │   ┌───────────────────┘
                     │   │
                     ▼   ▼
         ┌────────────────────────────┐
         │ Reinstalls in editable mode│
         │ .venv/bin/pip install -e . │
         └──────────────┬─────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ ✅ Fixed!     │
                │ Re-run test   │
                └───────────────┘
```

## When Do You Need to Run `make sync-deps`?

### ✅ Obvious Indicators (You'll know!)

**1. `make test` fails with clear message:**

```
❌ Environment out of sync - package not in editable mode

💡 Quick fix: Run one of these commands:
   make sync-deps           # Recommended: full sync
   make test AUTO_FIX=1     # Auto-fix and continue
```

↪️ **Action**: Run `make sync-deps` or use `AUTO_FIX=1`

**2. After pulling changes:**

```bash
git pull origin main
# New code added/changed in src/
```

↪️ **Action**: Run `make sync-deps` to pick up changes

**3. Mysterious test failures:**

```
Tests pass in CI but fail locally
Tests fail but source code looks correct
ImportError for recently added code
```

↪️ **Action**: Run `make check-venv` to diagnose

### ❌ You DON'T need to run it if:

- ✅ Tests are passing
- ✅ No recent `git pull`
- ✅ `make check-venv` shows ✓ all green

## CI Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     CI: Every run is fresh                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Fresh Python env      │
                │ No cache, no stale    │
                └───────┬───────────────┘
                        │
                        ▼
            ┌──────────────────────────────┐
            │ pip install -e ".[dev,all]"  │
            │ (Always editable mode)       │
            └────────────┬─────────────────┘
                         │
                         ▼
            ┌──────────────────────────────┐
            │ python scripts/sync-deps.py  │
            │ --check                      │
            └────┬─────────────────┬───────┘
                 │                 │
            PASS │                 │ FAIL
                 │                 │
                 ▼                 ▼
         ┌───────────┐      ┌─────────────────┐
         │ ✅ Tests  │      │ ❌ CI BUG!      │
         │ run       │      │ (not your code) │
         └───────────┘      └─────────────────┘
```

**Key Insight**: CI auto-syncs every run. If CI fails validation, it's a **workflow bug**, not your code.

## Comparison: Manual vs Auto-Fix

### Manual Flow (More Control)

```bash
make test
# ❌ Environment out of sync

make sync-deps    # You decide when to fix
make test         # Run again
# ✅ Tests pass
```

### Auto-Fix Flow (Faster)

```bash
make test AUTO_FIX=1
# 🔧 Auto-fixing...
# ✅ Fixed! Package reinstalled
# ✅ Tests pass
```

**When to use each**:

- **Manual**: When you want to review what's being fixed
- **Auto-fix**: When you just want tests to run (most common)

## Integration with Development Workflow

```
Day-to-day development:
┌──────────────────────────────────────────────────────────┐
│ 1. Write code                                            │
│ 2. make test              ← Auto-checks venv             │
│    ├─ If out of sync → make sync-deps or AUTO_FIX=1     │
│    └─ Tests run                                          │
│ 3. git commit             ← pre-commit hook runs         │
│ 4. git push               ← pre-push hook runs           │
│ 5. CI runs                ← Fresh env, auto-validates    │
└──────────────────────────────────────────────────────────┘

After pulling changes:
┌──────────────────────────────────────────────────────────┐
│ 1. git pull origin main                                  │
│ 2. make sync-deps         ← Proactive sync               │
│ 3. make test              ← Should pass immediately      │
└──────────────────────────────────────────────────────────┘

When joining project:
┌──────────────────────────────────────────────────────────┐
│ 1. git clone ...                                         │
│ 2. make dev-setup         ← One command sets up all     │
│ 3. make test              ← Should work immediately      │
└──────────────────────────────────────────────────────────┘
```

## Summary

**Q: How do I know I need to run `make sync-deps`?**

**A: You'll be told explicitly!**

- `make test` will **fail** with clear instructions
- Error message shows exactly what to run
- Can't miss it!

**Q: Can I just auto-fix?**

**A: Yes! Use `make test AUTO_FIX=1`**

- Fixes and continues in one command
- Safe operation (just reinstalls in editable mode)
- No risk of breaking anything

**Q: What about CI?**

**A: CI auto-syncs every run**

- Fresh environment each time
- No manual intervention needed
- If sync validation fails → it's a workflow bug (file an issue)

**Q: What's the "source of truth"?**

**A: Your source code in `src/`**

- Local venv syncs to source via editable install
- CI syncs to source via fresh editable install
- Everyone tests the same code
