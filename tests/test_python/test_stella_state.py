import struct

from env_stableretro_turbo.stella_state import migrate_legacy_state


def test_migrate_legacy_state_preserves_authority_state_bytes():
    legacy = (
        struct.pack("<I", len(b"03090100state"))
        + b"03090100state"
        + struct.pack("<I", len(b"System"))
        + b"System"
        + struct.pack("<I", 123)
        + b"\x7f"
        + b"remaining-state"
    )

    assert migrate_legacy_state(object(), legacy) is legacy


def test_migrate_legacy_state_leaves_current_state_unchanged():
    current = struct.pack("<I", len(b"03090101state")) + b"03090101state"
    assert migrate_legacy_state(object(), current) is current
