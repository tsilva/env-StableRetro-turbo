import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION_PATH = Path(__file__).resolve().parent / "stable_retro" / "VERSION.txt"
SCRIPT_DIR = Path(__file__).resolve().parent
README = (SCRIPT_DIR / "README.md").read_text(encoding="utf-8")


def is_macos_rosetta_process():
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-in", "sysctl.proc_translated"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return False
    return result.stdout.strip() == "1"


if is_macos_rosetta_process():
    raise RuntimeError(
        "stable-retro-turbo does not support Rosetta builds. "
        "Use a native arm64 Python on Apple Silicon.",
    )

PUBLIC_CORE_NAMES = tuple(
    core.strip()
    for core in os.environ.get(
        "STABLE_RETRO_PUBLIC_CORES",
        "gambatte,fceumm,snes9x,genesis_plus_gx,stella,mgba,picodrive,mednafen_saturn,melonds",
    ).split(",")
    if core.strip()
)
PUBLIC_DATA_PLATFORMS = frozenset(
    platform.strip()
    for platform in os.environ.get("STABLE_RETRO_PUBLIC_DATA_PLATFORMS", "").split(",")
    if platform.strip()
)

DATA_PACKAGE_DIRS = {
    "stable_retro.data.stable": SCRIPT_DIR / "stable_retro" / "data" / "stable",
    "stable_retro.data.experimental": SCRIPT_DIR
    / "stable_retro"
    / "data"
    / "experimental",
    "stable_retro.data.contrib": SCRIPT_DIR / "stable_retro" / "data" / "contrib",
}


def is_rom_payload(path: Path):
    return path.name.startswith("rom.") and path.name != "rom.sha"


def package_files(root: Path, *, exclude=None):
    files = []
    exclude = exclude or (lambda _: False)
    for path in sorted(root.rglob("*")):
        if path.is_file() and not exclude(path):
            files.append(path.relative_to(root).as_posix())
    return files


def data_dir_platform(name: str):
    parts = name.rsplit("-", 2)
    if len(parts) >= 3 and parts[-1].startswith("v"):
        return parts[-2]
    return name.rsplit("-", 1)[-1]


def is_excluded_data_platform(root: Path, path: Path):
    if not PUBLIC_DATA_PLATFORMS:
        return False
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        return False
    return data_dir_platform(relative.parts[0]) not in PUBLIC_DATA_PLATFORMS


def stable_retro_data_package_data(package_root: Path):
    return package_files(
        package_root,
        exclude=lambda path: is_rom_payload(path)
        or is_excluded_data_platform(package_root, path),
    )


def stable_retro_package_data():
    files = [
        "VERSION.txt",
        "README.md",
        "LICENSES.md",
    ]
    core_suffix = ".dylib" if sys.platform == "darwin" else ".so"
    if sys.platform == "win32":
        core_suffix = ".dll"
    for core in PUBLIC_CORE_NAMES:
        files.append(f"cores/{core}.json")
        files.append(f"cores/{core}_libretro{core_suffix}")
    return files


def copy_public_core_assets(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    core_suffix = ".dylib" if sys.platform == "darwin" else ".so"
    if sys.platform == "win32":
        core_suffix = ".dll"
    for core in PUBLIC_CORE_NAMES:
        for suffix in (".json", f"_libretro{core_suffix}"):
            asset_name = f"{core}{suffix}"
            asset = None
            direct_candidates = [
                destination / asset_name,
                SCRIPT_DIR / "stable_retro" / "cores" / asset_name,
            ]
            for candidate in direct_candidates:
                if candidate.exists():
                    asset = candidate
                    break
            if asset is None:
                for candidate in sorted((SCRIPT_DIR / "build").rglob(asset_name)):
                    if candidate.is_file():
                        asset = candidate
                        break
            if asset is None:
                raise FileNotFoundError(f"Missing public core asset: {asset_name}")
            target = destination / asset.name
            if asset.resolve() != target.resolve():
                shutil.copy2(asset, target)


def should_build_native_snes():
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and "snes9x" in PUBLIC_CORE_NAMES
    )


def macos_min_flag(default: str):
    min_version = os.environ.get("MACOSX_DEPLOYMENT_TARGET", default)
    return f"-mmacosx-version-min={min_version}"


def build_native_snes_core(destination: Path, jobs: str):
    source_dir = SCRIPT_DIR / "cores" / "snes" / "libretro"
    destination.mkdir(parents=True, exist_ok=True)
    min_flag = macos_min_flag("11.0")
    subprocess.check_call(["make", "-C", str(source_dir), "clean"])
    subprocess.check_call(
        [
            "make",
            "-C",
            str(source_dir),
            "platform=osx",
            "CC=clang -arch arm64",
            "CXX=clang++ -arch arm64",
            f"fpic=-fPIC {min_flag}",
            jobs,
        ],
    )
    shutil.copy2(
        source_dir / "snes9x_libretro.dylib",
        destination / "snes9x_libretro.dylib",
    )


class CMakeBuild(build_ext):
    def run(self):
        suffix = super().get_ext_filename("")
        pyext_suffix = f"-DPYEXT_SUFFIX={suffix}"
        package_dir = None
        if self.inplace:
            pylib_dir = f"-DPYLIB_DIRECTORY={SCRIPT_DIR}"
        else:
            ext_path = Path(self.get_ext_fullpath("stable_retro._retro"))
            package_dir = ext_path.parent
            pylib_dir = f"-DPYLIB_DIRECTORY={package_dir.parent}"
        # Provide hints to CMake about where to find Python (this should be enough for most cases)
        python_root_dir = f"-DPython_ROOT_DIR={os.path.dirname(sys.executable)}"
        python_find_strategy = "-DPython_FIND_STRATEGY=LOCATION"

        # These directly specify Python artifacts
        python_executable = f"-DPython_EXECUTABLE={sys.executable}"
        python_include_dir = f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}"
        python_library = f"-DPython_LIBRARY={sysconfig.get_path('platlib')}"

        # Allow users to pass extra CMake configure flags when building via pip.
        # Example:
        #   CMAKE_ARGS="-DBUILD_N64=OFF -DENABLE_HW_RENDER=ON" pip3 install -e .
        extra_cmake_args = []
        for env_var in ("STABLE_RETRO_CMAKE_ARGS", "CMAKE_ARGS"):
            value = os.environ.get(env_var)
            if value:
                extra_cmake_args.extend(shlex.split(value))
        has_build_type = any(
            arg == "-DCMAKE_BUILD_TYPE" or arg.startswith("-DCMAKE_BUILD_TYPE=")
            for arg in extra_cmake_args
        )
        if self.debug:
            build_type = "-DCMAKE_BUILD_TYPE=Debug"
        elif has_build_type:
            build_type = ""
        else:
            build_type = "-DCMAKE_BUILD_TYPE=Release"

        cmake_cmd = [
            "cmake",
            ".",
            "-G",
            "Unix Makefiles",
            build_type,
            pyext_suffix,
            pylib_dir,
            python_root_dir,
            python_find_strategy,
            python_executable,
            python_include_dir,
            python_library,
            *extra_cmake_args,
        ]

        subprocess.check_call([arg for arg in cmake_cmd if arg])
        if self.parallel:
            jobs = f"-j{self.parallel:d}"
        else:
            import multiprocessing

            jobs = f"-j{multiprocessing.cpu_count():d}"

        subprocess.check_call(["make", jobs, "all"])
        if package_dir is not None:
            copy_public_core_assets(package_dir / "cores")
            if should_build_native_snes():
                build_native_snes_core(package_dir / "cores", jobs)
        else:
            if should_build_native_snes():
                build_native_snes_core(SCRIPT_DIR / "stable_retro" / "cores", jobs)


setup(
    name="stable-retro-turbo",
    description="Blazing-fast Stable Retro fork with native vectorization and preprocessing",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Farama Foundation",
    author_email="contact@farama.org",
    url="https://github.com/tsilva/stable-retro-turbo",
    version=VERSION_PATH.read_text(encoding="utf-8").strip(),
    license="MIT",
    install_requires=[
        "gymnasium>=1.0.0",
        "pyglet>=1.5.27,<2",
        "farama-notifications>=0.0.1",
    ],
    python_requires=">=3.11,<3.15",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    ext_modules=[Extension("stable_retro._retro", ["CMakeLists.txt", "src/*.cpp"])],
    cmdclass={"build_ext": CMakeBuild},
    packages=[
        "retro",  # Compatibility shim
        "stable_retro",
        "stable_retro.data",
        "stable_retro.data.stable",
        "stable_retro.data.experimental",
        "stable_retro.data.contrib",
        "stable_retro.scripts",
        "stable_retro.import",
        "stable_retro.examples",
        "stable_retro.testing",
    ],
    package_data={
        "stable_retro": stable_retro_package_data(),
        **{
            package_name: stable_retro_data_package_data(package_root)
            for package_name, package_root in DATA_PACKAGE_DIRS.items()
        },
    },
)
