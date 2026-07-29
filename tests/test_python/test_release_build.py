import ast
import importlib.util
import tomllib
from pathlib import Path


def _release_build_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "release_build.py"
    spec = importlib.util.spec_from_file_location("release_build", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_dependency_bounds_match_the_supported_contract():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    install_requires = next(
        keyword.value
        for keyword in setup_call.keywords
        if keyword.arg == "install_requires"
    )

    assert ast.literal_eval(install_requires) == [
        "gymnasium>=1.1,<2",
        "numpy>=1.26,<3",
        "pyglet>=1.5.27,<2",
        "farama-notifications>=0.0.1",
    ]


def test_next_post_version_increments_existing_post_version():
    release_build = _release_build_module()

    assert release_build.next_post_version("1.0.0.post22") == "1.0.0.post23"


def test_next_post_version_handles_base_version():
    release_build = _release_build_module()

    assert release_build.next_post_version("1.0.0") == "1.0.0.post1"


def test_version_file_is_the_single_source_of_truth():
    root = Path(__file__).resolve().parents[2]
    version_path = root / "stable_retro" / "VERSION.txt"

    assert (root / "setup.py").read_text(encoding="utf-8").count(
        'stable_retro" / "VERSION.txt',
    ) == 1
    assert "../stable_retro/VERSION.txt" in (root / "docs" / "conf.py").read_text(
        encoding="utf-8",
    )
    assert "stable_retro/VERSION.txt" in (
        root / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")
    assert (root / "stable_retro" / "__init__.py").read_text(encoding="utf-8").count(
        "VERSION.txt",
    ) == 1
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
    assert release_build.should_ignore(Path(".ccache"))
    assert not release_build.should_ignore(Path(".codex/skills/build"))


def test_rom_payload_detection_is_path_scoped():
    release_build = _release_build_module()

    assert release_build.is_rom_payload(Path("stable_retro/data/stable/Foo/rom.nes"))
    assert release_build.is_rom_payload(Path("stable_retro/data/stable/Foo/rom.md"))
    assert not release_build.is_rom_payload(
        Path("stable_retro/data/stable/Foo/rom.sha"),
    )
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
    assert "stella" in release_build.PUBLIC_CORES
    assert "Atari2600" in release_build.PUBLIC_DATA_PLATFORMS.split(",")
    assert "atari2600" in release_build.MACOS_CMAKE_ARGS


def test_expected_linux_wheels_match_auditwheel_policy_tag():
    release_build = _release_build_module()

    names = {path.name for path in release_build.expected_linux_wheels("1.0.1.post11")}

    assert release_build.PYTHON_TAGS == ("cp314",)
    assert len(names) == len(release_build.PYTHON_TAGS)
    assert all("manylinux_2_27_x86_64.manylinux_2_28_x86_64" in name for name in names)


def test_package_and_wheel_metadata_target_python_314_only():
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    setup_source = (root / "setup.py").read_text(encoding="utf-8")

    assert pyproject["project"]["requires-python"] == ">=3.14,<3.15"
    assert pyproject["tool"]["cibuildwheel"]["build"] == "cp314-*"
    assert 'python_requires=">=3.14,<3.15"' in setup_source
    assert '"Programming Language :: Python :: 3.14"' in setup_source


def test_wheel_build_parallelizes_cores_without_rebuilding_native_snes():
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    cmake_source = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    setup_source = (root / "setup.py").read_text(encoding="utf-8")

    assert "cmake>=3.28" in pyproject["build-system"]["requires"]
    assert "JOB_SERVER_AWARE TRUE" in cmake_source
    assert "set(core_env_command env)" in cmake_source
    assert "function(prepend_compiler_launcher" in cmake_source
    assert "${CMAKE_C_COMPILER_LAUNCHER}" in cmake_source
    assert "CC=${core_c_compiler}" in cmake_source
    assert "CXX=${core_cxx_compiler}" in cmake_source
    assert "build_native_snes_core" not in setup_source
    assert '["lipo", str(asset), "-verify_arch", "arm64"]' in setup_source


def test_release_wheel_builds_use_persistent_ccache():
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )

    for platform_name in ("macos", "linux"):
        platform = pyproject["tool"]["cibuildwheel"][platform_name]
        assert "ccache" in platform["before-all"]
        assert "ccache --zero-stats" in platform["before-all"]
        assert "ccache --max-size 2G" in platform["before-all"]
        assert "-DCMAKE_C_COMPILER_LAUNCHER=ccache" in platform["environment"]
        assert "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" in platform["environment"]
        assert "ccache --show-stats" in platform["repair-wheel-command"]

    assert "actions/cache@v4" in workflow
    assert "path: .ccache" in workflow
    assert (
        "key: ccache-v1-${{ matrix.platform }}-${{ runner.arch }}-${{ github.sha }}"
        in workflow
    )
    assert "ccache-v1-${{ matrix.platform }}-${{ runner.arch }}-" in workflow
    assert "Install macOS system packages" not in workflow


def test_release_platform_targets_are_canonical():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )
    release_build = _release_build_module()

    assert release_build.RELEASE_PLATFORMS == (
        "macos-arm64",
        "linux-x86_64",
    )
    for platform_name in release_build.RELEASE_PLATFORMS:
        assert f"platform: {platform_name}" in workflow
    assert "platform: macos\n" not in workflow
    assert "platform: linux\n" not in workflow
    assert "runner: macos-15" in workflow


def test_built_wheel_smoke_exercises_exact_live_snapshot_replay():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "release_build.py").read_text(encoding="utf-8")

    for required in (
        "Dr88-FamiconIntro.nes",
        "supports_live_snapshots",
        "capture_snapshots",
        '"snapshots": [handles[0], handles[0]]',
        'restored_infos["start_source"]',
        "np.testing.assert_array_equal(expected, actual)",
    ):
        assert required in source


def test_release_publish_requires_full_python_and_cpp_suites():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )

    assert "test-python:" in workflow
    assert "xvfb-run -s '-screen 0 1024x768x24' pytest" in workflow
    assert "test-cpp:" in workflow
    assert "ctest --progress --verbose" in workflow
    publish = workflow.split("  publish:", maxsplit=1)[1]
    assert "- test-python" in publish
    assert "- test-cpp" in publish


def test_release_cache_paths_are_platform_scoped(tmp_path):
    release_build = _release_build_module()

    macos = release_build.macos_env(tmp_path)
    linux = release_build.linux_env(tmp_path)

    assert macos["CCACHE_DIR"] == str(tmp_path / ".ccache")
    assert macos["CCACHE_BASEDIR"] == str(tmp_path)
    assert macos["CCACHE_COMPILERCHECK"] == "content"
    assert macos["CCACHE_MAXSIZE"] == "2G"
    assert linux["CIBW_CONTAINER_ENGINE"] == (
        f"docker; create_args: --volume={(tmp_path / '.ccache').resolve()}:/ccache"
    )


def test_cleaning_wheel_outputs_preserves_compiler_cache(tmp_path, monkeypatch):
    release_build = _release_build_module()
    monkeypatch.setattr(release_build, "REPO_ROOT", tmp_path)
    cache_marker = tmp_path / ".ccache" / "cache-entry"
    cache_marker.parent.mkdir()
    cache_marker.write_text("cached", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "dist").mkdir()

    release_build.clean_output_paths("1.0.1.post27", "macos-arm64")

    assert cache_marker.read_text(encoding="utf-8") == "cached"
    assert not (tmp_path / "build").exists()
    assert not (tmp_path / "dist").exists()
