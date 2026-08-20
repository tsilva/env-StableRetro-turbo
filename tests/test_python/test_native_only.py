import importlib
import types

import pytest

import env_stableretro_turbo as retro


def test_retro_emulator_is_native_only(monkeypatch):
    monkeypatch.setenv("ENV_STABLERETRO_TURBO_FORCE_ROSETTA_SNES", "1")
    monkeypatch.setenv("ENV_STABLERETRO_TURBO_BUILD_ROSETTA_SNES", "1")

    assert retro.RetroEmulator is retro.NativeRetroEmulator


def test_rosetta_snes_module_is_not_available():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("env_stableretro_turbo.rosetta_snes")


def test_rosetta_process_guard_rejects_translated_process(monkeypatch):
    monkeypatch.setattr(retro.sys, "platform", "darwin")
    monkeypatch.setattr(
        retro.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="1\n"),
    )

    with pytest.raises(RuntimeError, match="does not support running under Rosetta"):
        retro._reject_rosetta_process()
