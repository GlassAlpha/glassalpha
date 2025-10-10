"""Tabular model wrappers."""

# Import model wrappers to trigger registration decorators
# These are conditional imports - they only work if dependencies are installed
try:
    from . import lightgbm  # noqa: F401
except ImportError:
    pass

try:
    from . import sklearn  # noqa: F401
except ImportError:
    pass

try:
    from . import xgboost  # noqa: F401
except ImportError:
    pass
