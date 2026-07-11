import gzip
import struct

import stable_retro
from stable_retro.scripts.import_path import _refresh_atari_start_state


def test_refresh_atari_start_state_uses_current_core(monkeypatch, tmp_path):
    legacy = (
        struct.pack("<I", len(b"03090100state"))
        + b"03090100state"
        + struct.pack("<I", len(b"System"))
        + b"System"
        + struct.pack("<I", 123)
        + b"\x7f"
        + b"remaining-state"
    )
    (tmp_path / "Start.state").write_bytes(gzip.compress(legacy, mtime=0))

    class FakeEmulator:
        def __init__(self, rom_path):
            assert rom_path == "breakout.a26"

        def set_state(self, state):
            assert b"03090101state" in state[:32]
            return False

        def get_state(self):
            return b"current-stella-state"

    monkeypatch.setattr(stable_retro, "RetroEmulator", FakeEmulator)

    _refresh_atari_start_state(
        "Breakout-Atari2600-v0",
        str(tmp_path),
        "breakout.a26",
    )

    assert gzip.decompress((tmp_path / "Start.state").read_bytes()) == (
        b"current-stella-state"
    )


def test_refresh_atari_start_state_ignores_other_platforms(tmp_path):
    _refresh_atari_start_state("Mario-Nes-v0", str(tmp_path), "mario.nes")
    assert not (tmp_path / "Start.state").exists()
