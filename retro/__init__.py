"""
Backward compatibility module for retro -> env_stableretro_turbo migration.

This module provides a compatibility layer that allows 'import retro' to continue
working while warning users about the deprecation.
"""

import importlib
import sys
import warnings

import env_stableretro_turbo as _env_stableretro_turbo
import env_stableretro_turbo._retro
import env_stableretro_turbo.data
import env_stableretro_turbo.enums
import env_stableretro_turbo.examples
import env_stableretro_turbo.retro_env
import env_stableretro_turbo.scripts
import env_stableretro_turbo.testing

# Import and re-export everything from env_stableretro_turbo
from env_stableretro_turbo import *  # noqa: F401, F403

# Issue deprecation warning (after imports to satisfy E402)
warnings.warn(
    "The 'retro' package name is deprecated and will be removed in a future version. "
    "Please use 'import env_stableretro_turbo' instead of 'import retro'.",
    DeprecationWarning,
    stacklevel=2,
)

env_stableretro_turbo_import = importlib.import_module("env_stableretro_turbo.import")

# Map the modules
sys.modules["retro"] = sys.modules["env_stableretro_turbo"]
sys.modules["retro.data"] = env_stableretro_turbo.data
sys.modules["retro.scripts"] = env_stableretro_turbo.scripts
sys.modules["retro.examples"] = env_stableretro_turbo.examples
sys.modules["retro.import"] = env_stableretro_turbo_import
sys.modules["retro.testing"] = env_stableretro_turbo.testing
sys.modules["retro._retro"] = env_stableretro_turbo._retro
sys.modules["retro.enums"] = env_stableretro_turbo.enums
sys.modules["retro.retro_env"] = env_stableretro_turbo.retro_env

# Try to import rendering if it exists.
# In headless environments pyglet can fail during module import even though the
# rest of the compatibility layer is usable (for example `python -m retro.import`).
try:
    import env_stableretro_turbo.rendering

    sys.modules["retro.rendering"] = env_stableretro_turbo.rendering
except Exception:
    pass

# Re-export commonly used items
__version__ = _env_stableretro_turbo.__version__
__all__ = _env_stableretro_turbo.__all__
