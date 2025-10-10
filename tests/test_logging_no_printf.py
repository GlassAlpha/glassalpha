"""Snapshot test to ban printf-style logging across the codebase.

This test enforces single-argument logging calls to prevent printf-style
logging regressions that caused CI test failures.

Contract:
- Every logger.* call must have exactly one positional argument
- Keyword args like exc_info=True are allowed
- This catches logger.info("... %s", thing) patterns
"""

import ast
import pathlib

import pytest

LOG_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}


def test_no_printf_style_logging_single_arg_only():
    """Ensure all logging calls use single-argument format (no printf-style)."""
    src_root = pathlib.Path(__file__).parents[1] / "src" / "glassalpha"
    offenders = []

    for path in src_root.rglob("*.py"):
        # Skip vendor or generated files if any (adjust as needed)
        try:
            code = path.read_text(encoding="utf-8")
            tree = ast.parse(code, filename=str(path))
        except (UnicodeDecodeError, SyntaxError):
            # Skip files that can't be parsed
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in LOG_METHODS:
                    # Allow keyword-only args like exc_info=True; ban multiple *positional* args
                    if len(node.args) != 1:
                        offenders.append(f"{path.relative_to(src_root.parent.parent)}:{node.lineno}")

    if offenders:
        msg = "printf-style logging detected (multiple positional args):\n" + "\n".join(offenders)
        msg += "\n\nUse f-strings instead:"
        msg += "\n  ✅ logger.info(f'Message: {value}')"
        msg += "\n  ❌ logger.info('Message: %s', value)"
        msg += "\n\nThis prevents test mock assertion failures."
        pytest.fail(msg)


def test_pipeline_init_message_exact_format():
    """Ensure pipeline init message uses exact expected format."""
    src_root = pathlib.Path(__file__).parents[1] / "src" / "glassalpha"
    pipeline_file = src_root / "pipeline" / "audit.py"

    if not pipeline_file.exists():
        pytest.skip("Pipeline file not found")

    code = pipeline_file.read_text(encoding="utf-8")

    # Check for exact init message pattern
    expected_patterns = [
        "log_pipeline_init(logger, config.audit_profile)",
        'logger.info("Initialized audit pipeline with profile: tabular_compliance")',
        'logger.info(f"Initialized audit pipeline with profile: {',
    ]

    has_correct_pattern = any(pattern in code for pattern in expected_patterns)
    assert has_correct_pattern, (
        "Pipeline init logging must use exact contract format. Use log_pipeline_init() helper or exact string format."
    )
