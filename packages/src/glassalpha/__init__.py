"""GlassAlpha: AI Compliance Toolkit for Regulated ML

Fast imports with lazy module loading (PEP 562).
"""

import sys

# Python version check - fail fast with clear message
if sys.version_info < (3, 11):
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    raise RuntimeError(
        f"GlassAlpha requires Python 3.11 or higher.\n"
        f"You have Python {current_version}.\n\n"
        f"To install Python 3.11.8:\n\n"
        f"1. Visit: https://python.org/downloads\n"
        f"2. Download Python 3.11.8 installer\n"
        f"3. Run installer and check 'Add Python to PATH'\n"
        f"4. Open new terminal and run: python --version\n\n"
        f"Alternative with pyenv (recommended for developers):\n"
        f"curl https://pyenv.run | bash\n"
        f"pyenv install 3.11.8\n"
        f"pyenv global 3.11.8\n\n"
        f"Need help? See: https://docs.python.org/3/using/index.html",
    )

import importlib
from typing import Any

__version__ = "0.2.0"

# Public API (lazy-loaded modules)
__all__ = [
    "__version__",
    "audit",
    "config",
    "datasets",
    "utils",
]

# Lazy module loading (PEP 562)
_LAZY_MODULES = {
    "audit": "glassalpha.api",  # Maps to glassalpha.api (contains from_model, etc.)
    "config": "glassalpha.config",  # Configuration loading and validation
    "datasets": "glassalpha.datasets",
    "utils": "glassalpha.utils",
}


def __getattr__(name: str) -> Any:
    """Lazy-load modules on first access (PEP 562)

    This enables fast imports by deferring heavy dependencies
    (sklearn, xgboost, matplotlib) until actually needed.

    Example:
        >>> import glassalpha as ga  # <200ms, no heavy deps
        >>> result = ga.audit.from_model(...)  # Now loads deps

    """
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'glassalpha' has no attribute '{name}'")


def __dir__():
    """Enable tab-completion for lazy modules"""
    return __all__
