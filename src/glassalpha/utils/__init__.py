"""Utilities package for GlassAlpha.

This package provides core utilities organized by responsibility:

**Module Responsibilities**:

- `manifest.py`: Audit manifest generation and metadata management (631 lines)
- `seeds.py`: Deterministic random seed management and component seeding
- `hashing.py`: Content hashing for configs, dataframes, files, and objects
- `determinism.py`: Core determinism utilities and validation helpers
- `determinism_validator.py`: Full determinism testing framework (runs multiple audits)
- `features.py`: Feature name normalization and validation
- `preprocessing.py`: Preprocessing artifact utilities
- `cache_dirs.py`: Cache directory management for datasets and artifacts
- `integrity.py`: Dataset integrity checks and verification
- `locks.py`: File locking utilities for concurrent access
- `progress.py`: Progress bar utilities for long-running operations

**Import Guidelines**:

Prefer importing from this package level for commonly used utilities:
    from glassalpha.utils import hash_dataframe, set_global_seed, AuditManifest

For specialized utilities, import from specific modules:
    from glassalpha.utils.determinism_validator import DeterminismValidator
    from glassalpha.utils.preprocessing import load_preprocessing_artifact
"""

from __future__ import annotations

# Feature utilities
from .features import align_features

# Hashing utilities
from .hashing import hash_config, hash_dataframe, hash_file, hash_object

# Core utilities (lightweight, no heavy dependencies)
from .manifest import AuditManifest, ManifestGenerator
from .seeds import (
    SeedManager,
    get_component_seed,
    get_seeds_manifest,
    set_global_seed,
    with_component_seed,
    with_seed,
)

# Public API - commonly used utilities
__all__ = [
    "AuditManifest",
    "ManifestGenerator",
    "SeedManager",
    "align_features",
    "get_component_seed",
    "get_seeds_manifest",
    "hash_config",
    "hash_dataframe",
    "hash_file",
    "hash_object",
    "set_global_seed",
    "with_component_seed",
    "with_seed",
]

# Note: Specialized utilities should be imported from their specific modules:
# - DeterminismValidator: from glassalpha.utils.determinism_validator
# - Preprocessing: from glassalpha.utils.preprocessing
# - Cache dirs: from glassalpha.utils.cache_dirs
# - Integrity checks: from glassalpha.utils.integrity
# - Progress bars: from glassalpha.utils.progress
