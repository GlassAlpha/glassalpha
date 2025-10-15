"""Audit entry points: from_model, from_predictions, from_config, run_audit.

Main API surface for generating audit results.

This module re-exports from focused submodules for backward compatibility.
Direct imports from submodules are also supported for explicit usage.

LAZY IMPORTS: numpy and pandas are imported inside functions to enable
basic module imports without scientific dependencies installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from glassalpha.api.from_config import from_config
from glassalpha.api.from_model import from_model
from glassalpha.api.from_predictions import from_predictions
from glassalpha.api.result import AuditResult
from glassalpha.api.run_audit import run_audit

# Re-export for backward compatibility
__all__ = [
    "AuditResult",
    "from_config",
    "from_model",
    "from_predictions",
    "run_audit",
]
