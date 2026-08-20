import gzip
import struct

import env_stableretro_turbo
from env_stableretro_turbo.scripts.import_path import _refresh_atari_start_state


def test_refresh_atari_start_state_preserves_authority_format(monkeypatch, tmp_path):
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

    monkeypatch.setattr(env_stableretro_turbo, "RetroEmulator", FakeEmulator)

    _refresh_atari_start_state(
        "Breakout-Atari2600-v0",
        str(tmp_path),
        "breakout.a26",
    )

    assert gzip.decompress((tmp_path / "Start.state").read_bytes()) == legacy


def test_refresh_atari_start_state_ignores_other_platforms(tmp_path):
    _refresh_atari_start_state("Mario-Nes-v0", str(tmp_path), "mario.nes")
    assert not (tmp_path / "Start.state").exists()
