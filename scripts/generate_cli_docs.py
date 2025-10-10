#!/usr/bin/env python3
"""Generate CLI documentation from Typer app.

This script extracts command help text from the Typer CLI app and generates
Markdown documentation. It ensures CLI docs stay in sync with actual commands.

Usage:
    python scripts/generate_cli_docs.py > site/docs/reference/cli.md
    python scripts/generate_cli_docs.py --check  # CI mode: fail if docs are stale
"""

import sys
from pathlib import Path
from typing import Any

import typer
from typer.main import get_command


def extract_command_help(app: typer.Typer | Any, prefix: str = "") -> list[dict[str, Any]]:
    """Extract help text from Typer app commands.

    Args:
        app: Typer application instance or Click command
        prefix: Command prefix for nested commands

    Returns:
        List of command documentation dictionaries

    """
    commands = []

    # Get Click command from Typer app (only if it's a Typer instance)
    if isinstance(app, typer.Typer):
        click_app = get_command(app)
    else:
        # Already a Click command (from recursive call)
        click_app = app

    # Extract commands
    if hasattr(click_app, "commands"):
        for name, cmd in click_app.commands.items():
            full_name = f"{prefix}{name}" if prefix else name

            # Extract command info
            cmd_info = {
                "name": full_name,
                "short_help": cmd.get_short_help_str(limit=100),
                "help": cmd.help or "No description available",
                "options": [],
                "arguments": [],
            }

            # Extract parameters (options and arguments)
            for param in cmd.params:
                param_info = {
                    "name": param.name,
                    "type": param.type.name if hasattr(param.type, "name") else str(param.type),
                    "help": param.help or "",
                    "required": param.required,
                    "default": str(param.default) if param.default is not None else None,
                }

                if isinstance(param, typer.core.TyperOption):
                    # Format option names
                    opts = param.opts or []
                    param_info["flags"] = ", ".join(opts)
                    cmd_info["options"].append(param_info)
                elif isinstance(param, typer.core.TyperArgument):
                    cmd_info["arguments"].append(param_info)

            commands.append(cmd_info)

            # Recursively extract subcommands if this is a group
            # Pass the Click command directly (not through get_command again)
            if hasattr(cmd, "commands"):
                subcommands = extract_command_help(cmd, prefix=f"{full_name} ")
                commands.extend(subcommands)

    return commands


def format_command_markdown(cmd: dict[str, Any]) -> str:
    """Format command info as Markdown.

    Args:
        cmd: Command information dictionary

    Returns:
        Markdown formatted string

    """
    lines = []

    # Command header
    lines.append(f"### `glassalpha {cmd['name']}`")
    lines.append("")
    lines.append(cmd["help"])
    lines.append("")

    # Arguments
    if cmd["arguments"]:
        lines.append("**Arguments:**")
        lines.append("")
        for arg in cmd["arguments"]:
            req = "required" if arg["required"] else "optional"
            lines.append(f"- `{arg['name']}` ({arg['type']}, {req}): {arg['help']}")
        lines.append("")

    # Options
    if cmd["options"]:
        lines.append("**Options:**")
        lines.append("")
        for opt in cmd["options"]:
            flags = opt.get("flags", opt["name"])
            default = f" (default: `{opt['default']}`)" if opt["default"] else ""
            lines.append(f"- `{flags}`: {opt['help']}{default}")
        lines.append("")

    # Examples (if in help text)
    if "Example:" in cmd["help"]:
        lines.append("**Example:**")
        lines.append("")
        lines.append("```bash")
        # Extract example from help text
        example_start = cmd["help"].find("Example:")
        example_text = cmd["help"][example_start:].split("\n")[1]
        lines.append(example_text.strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def generate_cli_docs() -> str:
    """Generate complete CLI documentation.

    Returns:
        Markdown documentation string

    """
    # Import CLI app
    try:
        from glassalpha.cli import app
    except ImportError as e:
        print(f"Error: Could not import GlassAlpha CLI: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract commands
    commands = extract_command_help(app)

    # Build documentation
    lines = [
        "# CLI Reference",
        "",
        "Complete command-line interface reference for GlassAlpha.",
        "",
        "## Installation",
        "",
        "```bash",
        "pip install glassalpha",
        "```",
        "",
        "## Quick Start",
        "",
        "```bash",
        "# Get help",
        "glassalpha --help",
        "",
        "# Run audit with config",
        "glassalpha audit --config config.yaml --out report.html",
        "",
        "# Validate config",
        "glassalpha validate config.yaml",
        "```",
        "",
        "## Commands",
        "",
        "GlassAlpha provides the following commands:",
        "",
    ]

    # Add command documentation
    for cmd in sorted(commands, key=lambda x: x["name"]):
        lines.append(format_command_markdown(cmd))

    # Add footer
    lines.extend(
        [
            "## Global Options",
            "",
            "These options are available for all commands:",
            "",
            "- `--help`: Show help message and exit",
            "- `--version`: Show version and exit",
            "",
            "## Exit Codes",
            "",
            "GlassAlpha uses standard exit codes:",
            "",
            "- `0`: Success",
            "- `1`: Validation failure or policy gate failure",
            "- `2`: Runtime error",
            "- `3`: Configuration error",
            "",
            "## Environment Variables",
            "",
            "- `PYTHONHASHSEED`: Set for deterministic execution (recommended: `42`)",
            "- `GLASSALPHA_CONFIG_DIR`: Override default config directory",
            "- `GLASSALPHA_CACHE_DIR`: Override default cache directory",
            "",
            "---",
            "",
            "*This documentation is automatically generated from the CLI code.*",
            "*Last updated: See git history for this file.*",
        ],
    )

    return "\n".join(lines)


def check_docs_current(generated_docs: str, docs_path: Path) -> bool:
    """Check if documentation file is current.

    Args:
        generated_docs: Newly generated documentation
        docs_path: Path to existing documentation file

    Returns:
        True if docs are current, False otherwise

    """
    if not docs_path.exists():
        print(f"Error: Documentation file not found: {docs_path}", file=sys.stderr)
        return False

    existing_docs = docs_path.read_text()

    # Normalize line endings for comparison
    generated_normalized = generated_docs.replace("\r\n", "\n")
    existing_normalized = existing_docs.replace("\r\n", "\n")

    return generated_normalized == existing_normalized


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate CLI documentation")
    parser.add_argument("--check", action="store_true", help="Check if docs are current (CI mode)")
    parser.add_argument("--output", type=Path, help="Output file path (default: stdout)")

    args = parser.parse_args()

    # Generate docs
    try:
        docs = generate_cli_docs()
    except Exception as e:
        print(f"Error generating documentation: {e}", file=sys.stderr)
        sys.exit(2)

    # Check mode
    if args.check:
        docs_path = Path("../site/docs/reference/cli.md")
        if check_docs_current(docs, docs_path):
            print("✓ CLI documentation is up to date")
            sys.exit(0)
        else:
            print("✗ CLI documentation is out of date", file=sys.stderr)
            print(f"Run: python scripts/generate_cli_docs.py --output {docs_path}", file=sys.stderr)
            sys.exit(1)

    # Output mode
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(docs)
        print(f"✓ Generated CLI documentation: {args.output}", file=sys.stderr)
    else:
        print(docs)


if __name__ == "__main__":
    main()
