import importlib.util
import inspect
from importlib import resources
from types import ModuleType


def get_module_source(module: ModuleType, pkg_fallback: str | None = None, filename: str | None = None) -> str:
    """Return source code for a loaded module, working for both source checkouts and installed wheels.
    Tries inspect.getsource first; falls back to importlib.resources if needed.
    """
    # Primary: inspect.getsource (works when .py is available)
    try:
        return inspect.getsource(module)
    except OSError:
        pass

    # Fallback via importlib.resources (for installed wheels)
    if pkg_fallback and filename:
        try:
            # Ensure the fallback refers to a *package* (has submodule_search_locations)
            spec = importlib.util.find_spec(pkg_fallback)
            if not spec or spec.submodule_search_locations is None:
                raise FileNotFoundError(f"{pkg_fallback!r} is not a package or not importable")

            return (resources.files(pkg_fallback) / filename).read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError, AttributeError, TypeError):
            # Normalize any resource resolution problems to FileNotFoundError for the test
            raise FileNotFoundError(f"Could not load source for module {module.__name__} via fallback")

    # If still unavailable, raise a helpful error
    raise FileNotFoundError(f"Could not load source for module {module.__name__}")
