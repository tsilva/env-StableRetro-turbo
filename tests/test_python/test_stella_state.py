import struct

from stable_retro.stella_state import migrate_legacy_state


def test_migrate_legacy_state_inserts_rng_and_snapshots_current_core():
    legacy = (
        struct.pack("<I", len(b"03090100state"))
        + b"03090100state"
        + struct.pack("<I", len(b"System"))
        + b"System"
        + struct.pack("<I", 123)
        + b"\x7f"
        + b"remaining-state"
    )

    class Emulator:
        loaded = None

        def set_state(self, state):
            self.loaded = state
            return False

        def get_state(self):
            return b"current-state"

    emulator = Emulator()
    assert migrate_legacy_state(emulator, legacy) == b"current-state"
    assert b"03090101state" in emulator.loaded[:32]
    marker = struct.pack("<I", len(b"System")) + b"System"
    rng_offset = emulator.loaded.index(marker) + len(marker) + 5
    assert struct.unpack_from("<I", emulator.loaded, rng_offset) == (543,)


def test_migrate_legacy_state_leaves_current_state_unchanged():
    current = struct.pack("<I", len(b"03090101state")) + b"03090101state"
    assert migrate_legacy_state(object(), current) is current
