import importlib.util
from pathlib import Path


def _release_build_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "release_build.py"
    spec = importlib.util.spec_from_file_location("release_build", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_next_post_version_increments_existing_post_version():
    release_build = _release_build_module()

    assert release_build.next_post_version("1.0.0.post22") == "1.0.0.post23"


def test_next_post_version_handles_base_version():
    release_build = _release_build_module()

    assert release_build.next_post_version("1.0.0") == "1.0.0.post1"


def test_should_ignore_root_build_but_not_skill_build_directory():
    release_build = _release_build_module()

    assert release_build.should_ignore(Path("build"))
    assert not release_build.should_ignore(Path(".codex/skills/build"))


def test_rom_payload_detection_is_path_scoped():
    release_build = _release_build_module()

    assert release_build.is_rom_payload(Path("stable_retro/data/stable/Foo/rom.nes"))
    assert release_build.is_rom_payload(Path("stable_retro/data/stable/Foo/rom.md"))
    assert not release_build.is_rom_payload(Path("stable_retro/data/stable/Foo/rom.sha"))
    assert not release_build.is_rom_payload(Path("docs/rom.nes"))
