"""Command-line interface for GlassAlpha.

Simplified to 3 core commands: audit, quickstart, doctor
"""

from .main import app

__all__ = [
    "app",
]


# Entry point for the CLI
def main():
    """Main CLI entry point."""
    app()
