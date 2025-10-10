"""AST-based regression test to prevent printf-style logging.

Uses static analysis to catch logger.info(..., ...) calls that
would fail contract tests requiring single-argument logging.
"""

import ast
from pathlib import Path

import pytest


class LoggingVisitor(ast.NodeVisitor):
    """AST visitor to detect printf-style logging calls."""

    def __init__(self) -> None:
        """Initialize the logging violations visitor."""
        self.violations: list[str] = []
        self.current_file: str = ""

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes to check for logging violations."""
        # Check for logger.info/debug/warning/error calls
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.attr, str)
            and node.func.attr.lower() in {"info", "debug", "warning", "error", "critical"}
            and len(node.args) > 1
        ):
            line_number = getattr(node, "lineno", "unknown")
            self.violations.append(
                f"{self.current_file}:{line_number} - Multi-argument logging call: {ast.unparse(node)}",
            )

        self.generic_visit(node)


def test_no_printf_style_logging_in_codebase() -> None:
    """Test that no printf-style logging exists in the codebase.

    Prevents regression to logger.info("message %s", value) calls
    that fail contract tests expecting single-argument calls.
    """
    visitor = LoggingVisitor()
    src_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha"

    # Skip certain files that might legitimately use multi-arg logging
    skip_patterns = {
        "_test",  # Test files
        "test_",  # Test files
        "__pycache__",  # Cache files
    }

    python_files = []
    for py_file in src_dir.rglob("*.py"):
        # Skip files with excluded patterns
        if any(pattern in str(py_file) for pattern in skip_patterns):
            continue
        python_files.append(py_file)

    # Process each Python file
    for py_file in python_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            visitor.current_file = str(py_file.relative_to(src_dir))
            visitor.visit(tree)

        except (SyntaxError, UnicodeDecodeError) as e:
            pytest.fail(f"Could not parse {py_file}: {e}")

    # Report violations
    if visitor.violations:
        violation_msg = "\n".join(visitor.violations)
        pytest.fail(
            f"Printf-style logging detected in codebase:\n{violation_msg}\n\n"
            "Use single-argument logging instead:\n"
            "  BAD:  logger.info('Message %s', value)\n"
            "  GOOD: logger.info(f'Message {value}')\n"
            "  GOOD: log_pipeline_init(logger, profile)\n",
        )


def test_logging_utils_compliance() -> None:
    """Test that logging utils module follows its own rules."""
    logging_utils_file = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "logging_utils.py"

    if not logging_utils_file.exists():
        pytest.skip("logging_utils.py not found")

    content = logging_utils_file.read_text(encoding="utf-8")
    tree = ast.parse(content)

    visitor = LoggingVisitor()
    visitor.current_file = "logging_utils.py"
    visitor.visit(tree)

    # logging_utils should never use printf-style logging
    assert not visitor.violations, f"logging_utils.py contains printf-style logging: {visitor.violations}"  # noqa: S101


def test_pipeline_audit_compliance() -> None:
    """Test that pipeline audit module uses single-arg logging."""
    pipeline_file = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "pipeline" / "audit.py"

    if not pipeline_file.exists():
        pytest.skip("pipeline/audit.py not found")

    content = pipeline_file.read_text(encoding="utf-8")
    tree = ast.parse(content)

    visitor = LoggingVisitor()
    visitor.current_file = "pipeline/audit.py"
    visitor.visit(tree)

    # Pipeline should use centralized logging helpers
    assert not visitor.violations, f"pipeline/audit.py contains printf-style logging: {visitor.violations}"  # noqa: S101


def test_ast_visitor_catches_violations() -> None:
    """Test that the AST visitor correctly identifies violations."""
    # Test code with printf-style logging
    bad_code = """
import logging
logger = logging.getLogger(__name__)

def bad_function():
    logger.info("Message %s with %d args", "test", 42)
    logger.debug("Debug %s", "message")
"""

    tree = ast.parse(bad_code)
    visitor = LoggingVisitor()
    visitor.current_file = "test.py"
    visitor.visit(tree)

    # Should detect 2 violations
    assert len(visitor.violations) == 2  # noqa: PLR2004, S101
    assert "Multi-argument logging call" in visitor.violations[0]  # noqa: S101
    assert "Multi-argument logging call" in visitor.violations[1]  # noqa: S101


def test_ast_visitor_allows_single_arg() -> None:
    """Test that single-argument logging is allowed."""
    good_code = """
import logging
logger = logging.getLogger(__name__)

def good_function():
    logger.info("Single argument message")
    logger.info(f"Formatted {message}")
    logger.debug("Another single arg")
"""

    tree = ast.parse(good_code)
    visitor = LoggingVisitor()
    visitor.current_file = "test.py"
    visitor.visit(tree)

    # Should detect no violations
    assert len(visitor.violations) == 0  # noqa: S101
