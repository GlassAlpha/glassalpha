"""Profile registry for GlassAlpha.

This module defines the ProfileRegistry using the decorator-friendly registry
and provides utilities for profile management.
"""

import logging

from ..core.decor_registry import DecoratorFriendlyRegistry

logger = logging.getLogger(__name__)

# Create the profile registry using the decorator-friendly registry
ProfileRegistry = DecoratorFriendlyRegistry(group="glassalpha.profiles")
ProfileRegistry.discover()  # Safe discovery without heavy imports
