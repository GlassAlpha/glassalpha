"""GlassAlpha API: Public audit interface.

Exports audit entry points and result classes.
"""

from glassalpha.api.audit import from_config, from_model, from_predictions, run_audit
from glassalpha.api.result import AuditResult

__all__ = [
    "AuditResult",
    "from_config",
    "from_model",
    "from_predictions",
    "run_audit",
]
