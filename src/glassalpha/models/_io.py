"""Stub for test compatibility - model I/O removed.

This module was removed during simplification but some tests still import it.
"""


def read_wrapper_state(path):
    """Stub for test compatibility."""
    raise NotImplementedError("Model I/O system was simplified - use joblib directly")


def write_wrapper_state(wrapper, path):
    """Stub for test compatibility."""
    raise NotImplementedError("Model I/O system was simplified - use joblib directly")


__all__ = ["read_wrapper_state", "write_wrapper_state"]
