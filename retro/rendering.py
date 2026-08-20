"""Compatibility shim for ``retro.rendering``.

Importing this module forwards to ``env_stableretro_turbo.rendering`` without forcing
the renderer to initialize during ``import retro``.
"""

from env_stableretro_turbo.rendering import *  # noqa: F401, F403
