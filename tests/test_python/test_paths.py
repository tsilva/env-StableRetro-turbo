import hashlib
import os
import sys
import types

import pytest

import stable_retro as retro


@pytest.fixture
def custom_cleanup():
    retro.data.Integrations.clear_custom_paths()
    assert not retro.data.Integrations.CUSTOM_ONLY.paths

    yield

    retro.data.Integrations.clear_custom_paths()
    assert not retro.data.Integrations.CUSTOM_ONLY.paths


def test_basic_paths():
    assert retro.data.Integrations.STABLE.paths == ["stable"]
    assert retro.data.Integrations.CONTRIB_ONLY.paths == ["contrib"]
    assert retro.data.Integrations.EXPERIMENTAL_ONLY.paths == ["experimental"]
    assert not retro.data.Integrations.CUSTOM_ONLY.paths

    assert retro.data.Integrations.CONTRIB.paths == ["contrib", "stable"]
    assert retro.data.Integrations.EXPERIMENTAL.paths == ["experimental", "stable"]
    assert retro.data.Integrations.CUSTOM.paths == ["stable"]

    assert retro.data.Integrations.ALL.paths == ["contrib", "experimental", "stable"]


def test_custom_path(custom_cleanup):
    assert not retro.data.Integrations.CUSTOM_ONLY.paths
    assert retro.data.Integrations.CUSTOM.paths == ["stable"]

    retro.data.Integrations.add_custom_path("a")
    assert retro.data.Integrations.CUSTOM_ONLY.paths == ["a"]
    assert retro.data.Integrations.CUSTOM.paths == ["a", "stable"]

    retro.data.Integrations.add_custom_path("b")
    assert retro.data.Integrations.CUSTOM_ONLY.paths == ["a", "b"]
    assert retro.data.Integrations.CUSTOM.paths == ["a", "b", "stable"]


def test_custom_path_default(custom_cleanup):
    assert not retro.data.Integrations.CUSTOM_ONLY.paths
    assert retro.data.Integrations.CUSTOM.paths == ["stable"]
    assert retro.data.Integrations.DEFAULT.paths == ["stable"]

    retro.data.add_custom_integration("a")
    assert retro.data.Integrations.CUSTOM_ONLY.paths == ["a"]
    assert retro.data.Integrations.CUSTOM.paths == ["a", "stable"]
    assert retro.data.Integrations.DEFAULT.paths == ["a", "stable"]

    retro.data.DefaultIntegrations.reset()
    assert retro.data.Integrations.CUSTOM_ONLY.paths == ["a"]
    assert retro.data.Integrations.CUSTOM.paths == ["a", "stable"]
    assert retro.data.Integrations.DEFAULT.paths == ["stable"]


def test_custom_path_absolute(custom_cleanup):
    assert not retro.data.get_file_path(
        "",
        "Dekadence-Dekadrive.md",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )

    test_rom_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../roms")
    retro.data.Integrations.add_custom_path(test_rom_dir)
    assert retro.data.get_file_path(
        "",
        "Dekadence-Dekadrive.md",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    ) == os.path.join(test_rom_dir, "Dekadence-Dekadrive.md")


def test_custom_path_relative(custom_cleanup):
    assert not retro.data.get_file_path(
        "Airstriker-Genesis",
        "rom.md",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )

    retro.data.Integrations.add_custom_path(retro.data.Integrations.STABLE.paths[0])
    assert retro.data.get_file_path(
        "Airstriker-Genesis",
        "rom.md",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    ) == retro.data.get_file_path(
        "Airstriker-Genesis",
        "rom.md",
        inttype=retro.data.Integrations.STABLE,
    )


def _install_fake_ale_roms(monkeypatch, roms):
    ale_module = types.ModuleType("ale_py")
    roms_module = types.ModuleType("ale_py.roms")

    def get_all_rom_ids():
        return list(roms)

    def get_rom_path(name):
        return roms.get(name)

    roms_module.get_all_rom_ids = get_all_rom_ids
    roms_module.get_rom_path = get_rom_path
    ale_module.roms = roms_module
    monkeypatch.setitem(sys.modules, "ale_py", ale_module)
    monkeypatch.setitem(sys.modules, "ale_py.roms", roms_module)


def test_atari_bin_extension_is_supported():
    assert retro.data.EMU_EXTENSIONS[".bin"] == "Atari2600"


def test_atari_rom_can_resolve_from_ale_py(monkeypatch, tmp_path):
    rom_path = tmp_path / "pong.bin"
    rom_path.write_bytes(b"fake pong")
    sha_path = retro.data.get_file_path("Pong-Atari2600-v0", "rom.sha")
    expected_sha = hashlib.sha1(rom_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        retro.data,
        "get_file_path",
        lambda game, file, *args, **kwargs: (
            sha_path if game == "Pong-Atari2600-v0" and file == "rom.sha" else None
        ),
    )
    monkeypatch.setattr(retro.data, "EMU_EXTENSIONS", {".a26": "Atari2600"})
    _install_fake_ale_roms(monkeypatch, {"pong": rom_path})
    monkeypatch.setattr(
        retro.data,
        "_expected_rom_shas",
        lambda game, inttype: {expected_sha},
    )

    assert retro.data.get_romfile_path("Pong-Atari2600-v0") == str(rom_path)


def test_atari_rom_rejects_ale_py_hash_mismatch(monkeypatch, tmp_path):
    rom_path = tmp_path / "pong.bin"
    rom_path.write_bytes(b"wrong pong")
    sha_path = retro.data.get_file_path("Pong-Atari2600-v0", "rom.sha")

    monkeypatch.setattr(
        retro.data,
        "get_file_path",
        lambda game, file, *args, **kwargs: (
            sha_path if game == "Pong-Atari2600-v0" and file == "rom.sha" else None
        ),
    )
    monkeypatch.setattr(retro.data, "EMU_EXTENSIONS", {".a26": "Atari2600"})
    _install_fake_ale_roms(monkeypatch, {"pong": rom_path})
    monkeypatch.setattr(
        retro.data,
        "_expected_rom_shas",
        lambda game, inttype: {"not-the-fake-rom-sha"},
    )

    with pytest.raises(FileNotFoundError):
        retro.data.get_romfile_path("Pong-Atari2600-v0")


def test_imported_atari_rom_takes_precedence_over_ale_py(
    custom_cleanup,
    monkeypatch,
    tmp_path,
):
    game_dir = tmp_path / "Pong-Atari2600-v0"
    game_dir.mkdir()
    imported_rom_path = game_dir / "rom.bin"
    imported_rom_path.write_bytes(b"imported pong")
    (game_dir / "rom.sha").write_text("unused\n")

    ale_rom_path = tmp_path / "ale-pong.bin"
    ale_rom_path.write_bytes(b"ale pong")
    _install_fake_ale_roms(monkeypatch, {"pong": ale_rom_path})
    retro.data.Integrations.add_custom_path(str(tmp_path))

    assert (
        retro.data.get_romfile_path(
            "Pong-Atari2600-v0",
            retro.data.Integrations.CUSTOM_ONLY,
        )
        == str(imported_rom_path)
    )
