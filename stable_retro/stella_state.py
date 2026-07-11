"""Compatibility helpers for Stella save-state revisions."""

from __future__ import annotations

import struct


LEGACY_HEADER = b"03090100state"
CURRENT_HEADER = b"03090101state"
_SYSTEM_MARKER = struct.pack("<I", len(b"System")) + b"System"
_DEFAULT_RANDOM_STATE = 543


def migrate_legacy_state(emulator, state: bytes) -> bytes:
    """Migrate a curated Stella 3.9.1 state into the current core format.

    The current core retains the legacy loader layout but additionally stores
    Stella's RNG after the System cycle counter and data-bus byte. Loading the
    adjusted legacy payload restores the curated gameplay position; taking a
    fresh snapshot then serializes every field added by the current core.
    """
    if state[4 : 4 + len(CURRENT_HEADER)] == CURRENT_HEADER:
        return state
    if state[4 : 4 + len(LEGACY_HEADER)] != LEGACY_HEADER:
        return state

    converted = bytearray(state)
    converted[4 : 4 + len(LEGACY_HEADER)] = CURRENT_HEADER
    marker_offset = converted.index(_SYSTEM_MARKER)
    rng_offset = marker_offset + len(_SYSTEM_MARKER) + 4 + 1
    converted[rng_offset:rng_offset] = struct.pack(
        "<I",
        _DEFAULT_RANDOM_STATE,
    )

    # The legacy payload ends before fields introduced by the newer core, so
    # the loader reports false after restoring the shared state. Snapshotting
    # immediately completes the migration in the current serialization format.
    emulator.set_state(bytes(converted))
    return emulator.get_state()
