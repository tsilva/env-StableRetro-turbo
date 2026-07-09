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


def test_version_file_is_the_single_source_of_truth():
    root = Path(__file__).resolve().parents[2]
    version_path = root / "stable_retro" / "VERSION.txt"

    assert (root / "setup.py").read_text(encoding="utf-8").count("stable_retro\" / \"VERSION.txt") == 1
    assert "../stable_retro/VERSION.txt" in (root / "docs" / "conf.py").read_text(encoding="utf-8")
    assert "stable_retro/VERSION.txt" in (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert (root / "stable_retro" / "__init__.py").read_text(encoding="utf-8").count("VERSION.txt") == 1
    assert version_path.read_text(encoding="utf-8").strip()


def test_latest_non_yanked_pypi_version_ignores_fully_yanked_latest_release():
    release_build = _release_build_module()
    releases = {
        "1.0.1.post3": [{"filename": "older.whl", "yanked": False}],
        "1.0.1.post4": [{"filename": "current.whl", "yanked": False}],
        "1.3.0": [
            {"filename": "macos.whl", "yanked": True},
            {"filename": "linux.whl", "yanked": True},
        ],
    }

    assert release_build.latest_non_yanked_pypi_version(releases) == "1.0.1.post4"


def test_latest_non_yanked_pypi_version_accepts_release_with_any_non_yanked_file():
    release_build = _release_build_module()
    releases = {
        "1.0.1.post4": [{"filename": "current.whl", "yanked": False}],
        "1.0.1.post5": [
            {"filename": "bad-platform.whl", "yanked": True},
            {"filename": "good-platform.whl", "yanked": False},
        ],
    }

    assert release_build.latest_non_yanked_pypi_version(releases) == "1.0.1.post5"


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


def test_public_native_macos_core_set_includes_packaged_arm64_cores():
    release_build = _release_build_module()

    assert {
        "mgba",
        "picodrive",
        "mednafen_saturn",
        "melonds",
    }.issubset(release_build.PUBLIC_CORES)
    for platform in ("GbAdvance", "32x", "Saturn", "NintendoDs"):
        assert platform in release_build.PUBLIC_DATA_PLATFORMS.split(",")
    for platform in ("gba", "32x", "saturn", "ds"):
        assert platform in release_build.MACOS_CMAKE_ARGS


def test_expected_linux_wheels_match_auditwheel_policy_tag():
    release_build = _release_build_module()

    names = {path.name for path in release_build.expected_linux_wheels("1.0.1.post11")}

    assert len(names) == len(release_build.PYTHON_TAGS)
    assert all("manylinux_2_27_x86_64.manylinux_2_28_x86_64" in name for name in names)
