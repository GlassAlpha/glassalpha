"""Utilities package for GlassAlpha.

Provides various utility functions and classes for:
- Manifest generation and management
- Data hashing and integrity checks
- Seed management for reproducibility
"""

from __future__ import annotations

# Import lightweight modules directly
from .manifest import AuditManifest, ManifestGenerator

# Conditional imports for modules with heavy dependencies
try:
    from .hashing import hash_config, hash_dataframe, hash_file, hash_object

    _HASHING_AVAILABLE = True
except ImportError:
    _HASHING_AVAILABLE = False

try:
    from .seeds import (
        SeedManager,
        get_component_seed,
        get_seeds_manifest,
        set_global_seed,
        with_component_seed,
        with_seed,
    )

    _SEEDS_AVAILABLE = True
except ImportError:
    _SEEDS_AVAILABLE = False

# Public API
__all__ = [
    "AuditManifest",
    "ManifestGenerator",
]

if _HASHING_AVAILABLE:
    __all__ += [
        "hash_config",
        "hash_dataframe",
        "hash_file",
        "hash_object",
    ]

if _SEEDS_AVAILABLE:
    __all__ += [
        "SeedManager",
        "get_component_seed",
        "get_seeds_manifest",
        "set_global_seed",
        "with_component_seed",
        "with_seed",
    ]
