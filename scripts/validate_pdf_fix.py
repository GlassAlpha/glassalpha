#!/usr/bin/env python3
"""Validation script for PDF generation timeout fix.

This script verifies that the PDF generation timeout fixes are properly implemented.
"""

import sys
from pathlib import Path


def check_cli_progress_callback():
    """Verify CLI passes progress callback to render_audit_pdf."""
    cli_file = Path("src/glassalpha/cli/commands.py")

    if not cli_file.exists():
        print("❌ CLI file not found")
        return False

    content = cli_file.read_text()

    # Check that progress_callback is passed (not None)
    if "progress_callback=pdf_progress_relay" not in content:
        print("❌ CLI doesn't pass progress_callback to render_audit_pdf")
        return False

    # Check that progress_queue is defined before worker
    if content.index("progress_queue =") > content.index("def pdf_generation_worker"):
        print("❌ progress_queue defined after worker function")
        return False

    print("✅ CLI progress callback is properly configured")
    return True


def check_timeout_enforcement():
    """Verify timeout enforcement in CLI monitoring loop."""
    cli_file = Path("src/glassalpha/cli/commands.py")
    content = cli_file.read_text()

    checks = [
        ("pdf_start_time = time.time()", "Start time tracking"),
        ("timeout_seconds = 300", "Timeout limit defined"),
        ("if elapsed > timeout_seconds:", "Overall timeout check"),
        ("if time.time() - last_progress_time > 90:", "Stall detection"),
        ("last_progress_time = time.time()", "Progress time tracking"),
    ]

    all_passed = True
    for check, description in checks:
        if check not in content:
            print(f"❌ Missing: {description}")
            all_passed = False

    if all_passed:
        print("✅ Timeout enforcement properly implemented")

    return all_passed


def check_pytest_timeout():
    """Verify pytest-timeout configuration."""
    pytest_ini = Path("pytest.ini")
    pyproject_toml = Path("pyproject.toml")

    if not pytest_ini.exists():
        print("❌ pytest.ini not found")
        return False

    if not pyproject_toml.exists():
        print("❌ pyproject.toml not found")
        return False

    pytest_content = pytest_ini.read_text()
    pyproject_content = pyproject_toml.read_text()

    checks_passed = True

    # Check pytest.ini
    if "--timeout=180" not in pytest_content:
        print("❌ pytest.ini missing --timeout=180")
        checks_passed = False

    if "--timeout-method=thread" not in pytest_content:
        print("❌ pytest.ini missing --timeout-method=thread")
        checks_passed = False

    # Check pyproject.toml
    if "pytest-timeout" not in pyproject_content:
        print("❌ pyproject.toml missing pytest-timeout dependency")
        checks_passed = False

    if checks_passed:
        print("✅ pytest-timeout configuration is correct")

    return checks_passed


def check_changelog():
    """Verify CHANGELOG documents the fix."""
    changelog = Path("CHANGELOG.md")

    if not changelog.exists():
        print("❌ CHANGELOG.md not found")
        return False

    content = changelog.read_text()

    keywords = [
        "PDF Generation Timeout Protection",
        "progress_callback",
        "timeout enforcement",
        "pytest-timeout",
    ]

    all_found = True
    for keyword in keywords:
        if keyword not in content:
            print(f"❌ CHANGELOG missing: {keyword}")
            all_found = False

    if all_found:
        print("✅ CHANGELOG properly documents the fix")

    return all_found


def main():
    """Run all validation checks."""
    print("🔍 Validating PDF generation timeout fix...\n")

    checks = [
        ("CLI Progress Callback", check_cli_progress_callback),
        ("Timeout Enforcement", check_timeout_enforcement),
        ("pytest-timeout Config", check_pytest_timeout),
        ("CHANGELOG Update", check_changelog),
    ]

    results = []
    for name, check_func in checks:
        print(f"\nChecking: {name}")
        print("-" * 50)
        passed = check_func()
        results.append((name, passed))

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\n✅ All validation checks passed!")
        print("\nNext steps:")
        print("  1. Install pytest-timeout: pip install -e '.[dev]'")
        print("  2. Run tests: make test")
        print("  3. Verify no hanging: timeout should occur in <3 minutes")
        return 0
    else:
        print("\n❌ Some validation checks failed")
        print("\nPlease review the failed checks above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
