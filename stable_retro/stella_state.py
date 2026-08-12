"""Compatibility helper retained for callers of older Turbo releases."""

from __future__ import annotations

LEGACY_HEADER = b"03090100state"
CURRENT_HEADER = b"03090101state"


def migrate_legacy_state(emulator, state: bytes) -> bytes:
    """Return an authority-format Stella 3.9.1 state unchanged.

    Stable Retro Turbo now deliberately ships the same Stella core and state
    format as the pinned Stable Retro semantic authority. The function remains
    as a no-op so downstream import tooling written for earlier Turbo releases
    does not need a flag day.
    """

    del emulator
    return state
