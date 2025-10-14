#!/usr/bin/env python3
"""Validate GlassAlpha example notebooks for QA testing.

Usage:
    python scripts/validate_notebooks.py [notebook_path]

If no path is provided, validates all notebooks in examples/notebooks/
"""

import json
import sys
from pathlib import Path


def validate_notebook(notebook_path: Path) -> dict:
    """Validate a single notebook.

    Returns:
        dict with keys: valid, code_cells, has_import, errors
    """
    result = {"path": str(notebook_path), "valid": False, "code_cells": 0, "has_glassalpha_import": False, "errors": []}

    try:
        with open(notebook_path) as f:
            nb = json.load(f)

        # Check basic structure
        if "cells" not in nb:
            result["errors"].append("No cells in notebook")
            return result

        # Count code cells
        code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
        result["code_cells"] = len(code_cells)

        # Check for GlassAlpha import (improved logic)
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            # Check for any form of glassalpha import
            if "import glassalpha" in source or "from glassalpha" in source:
                result["has_glassalpha_import"] = True
                break

        if not result["has_glassalpha_import"]:
            result["errors"].append("No GlassAlpha import found")

        # Mark as valid if no errors
        result["valid"] = len(result["errors"]) == 0

    except Exception as e:
        result["errors"].append(f"Exception: {e!s}")

    return result


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Validate specific notebook
        notebook_path = Path(sys.argv[1])
        notebooks = [notebook_path] if notebook_path.exists() else []
    else:
        # Validate all notebooks in examples/notebooks/
        notebooks_dir = Path("examples/notebooks")
        notebooks = list(notebooks_dir.glob("*.ipynb"))

    if not notebooks:
        print("❌ No notebooks found")
        sys.exit(1)

    all_valid = True
    for nb_path in sorted(notebooks):
        result = validate_notebook(nb_path)

        status = "✅" if result["valid"] else "❌"
        print(f"{status} {nb_path.name}")
        print(f"   Code cells: {result['code_cells']}")
        print(f"   GlassAlpha import: {result['has_glassalpha_import']}")

        if result["errors"]:
            for error in result["errors"]:
                print(f"   ⚠️  {error}")
            all_valid = False

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
