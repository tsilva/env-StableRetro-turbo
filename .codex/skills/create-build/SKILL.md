---
name: create-build
description: Build and validate the next stable-retro-turbo post-version release wheels. Use when the user says /create-build, asks to do a new build with the next version, or asks for macOS and Linux stable_retro_turbo wheels for publishing with twine.
---

# Create Build

## Overview

Use this skill to create publish-ready `stable_retro_turbo` wheels for the next `.postN` version on macOS arm64 and Linux x86_64. Treat `stable_retro/VERSION.txt` as the source of truth for the current version.

The expected final output is two wheel paths, SHA256 hashes, validation results, notable warnings, and the exact deploy command for uploading the built wheels with `twine`.

## Version And Worktree

1. Read `stable_retro/VERSION.txt` and bump to the next post version with `apply_patch`. For example, `1.0.0.post13` becomes `1.0.0.post14`.
2. Confirm the distribution name before building:

```bash
.venv314/bin/python setup.py --name
```

It must print `stable-retro-turbo`. The import package remains `stable_retro`, but the publish target and wheel basename must be `stable_retro_turbo`.

3. Run `git status --short --branch` before building. Preserve unrelated user changes and do not create or switch branches.
4. If the source changes in the current turn have not been regression-tested, run focused tests before building:

```bash
MPLCONFIGDIR=/tmp/matplotlib-stable-retro PYTHONPATH=. .venv314/bin/python -m pytest tests/test_python/test_vec_env.py -q
```

From `tests/`, run the NES core tests when emulator behavior changed:

```bash
./test-emulator --gtest_filter='EmulatorCore/EmulatorTest.*/*Nes'
```

## Clean Build Copies

Always build from clean source copies under `/private/tmp`, not directly from the working tree.

1. Create a unique temp root such as `/private/tmp/stable-retro-turbo-post<N>-builds.XXXXXX`.
2. Create `macos-src` and `linux-src-clean` copies with `rsync`.
3. Exclude stale build outputs and machine-local state:

```text
.git
.venv314
build
dist
CMakeCache.txt
CMakeFiles
wheelhouse*
*.o
*.a
*.so
*.dylib
*.d
stable_retro/_retro*.so
stable_retro/data/stable/*/rom.nes
stable_retro/data/stable/*/rom.sfc
stable_retro/data/stable/*/rom.smc
stable_retro/data/stable/*/rom.gb
stable_retro/data/stable/*/rom.gbc
stable_retro/data/stable/*/rom.md
stable_retro/data/stable/*/rom.gen
stable_retro/data/stable/*/rom.sms
stable_retro/data/stable/*/rom.bin
__pycache__
.pytest_cache
```

The `dist` and root `CMakeCache.txt` exclusions are important. Previous builds surfaced stale wheels and root CMake cache contamination when these were copied.

Exclude actual ROM payloads copied in during local ROM testing. Keep `rom.sha` metadata; it is part of the game metadata and does not contain the ROM. Also exclude compiled object, static library, and shared library artifacts from any prior local build. Copying macOS object files into `linux-src-clean` can break the Linux wheel build with errors like `file format not recognized`.

Before starting the Linux build, verify the clean copy has no copied compiled artifacts or ROM payloads:

```bash
find /private/tmp/<build-root>/linux-src-clean \( -name '*.o' -o -name '*.a' -o -name '*.so' -o -name '*.dylib' -o -name '*.d' \) | wc -l
find /private/tmp/<build-root>/linux-src-clean/stable_retro/data/stable -type f \( -name 'rom.nes' -o -name 'rom.sfc' -o -name 'rom.smc' -o -name 'rom.gb' -o -name 'rom.gbc' -o -name 'rom.md' -o -name 'rom.gen' -o -name 'rom.sms' -o -name 'rom.bin' \) | wc -l
```

Both commands should print `0`.

## Output Directories

Write final wheels to repository-level wheelhouses named for the post version:

```text
wheelhouse-post<N>-repaired
wheelhouse-post<N>-linux
```

Remove or avoid only artifacts you created for the current build. Do not delete user files or unrelated wheelhouses.

Use the repository venv by absolute path when the working directory is a clean source copy:

```bash
PY=/Users/tsilva/repos/tsilva/stable-retro-turbo/.venv314/bin/python
REPO=/Users/tsilva/repos/tsilva/stable-retro-turbo
```

## macOS Arm64 Wheel

Build on macOS arm64 using `$PY`, with the clean source copy as the working directory.

Use these environment values:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0
ARCHFLAGS='-arch arm64'
CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release -DBUILD_CORES=gb;nes;snes;genesis -DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF -DSTABLE_RETRO_USE_SYSTEM_LIBZIP=OFF'
STABLE_RETRO_PUBLIC_CORES='gambatte,fceumm,snes9x,genesis_plus_gx'
STABLE_RETRO_PUBLIC_DATA_PLATFORMS='GameBoy,Nes,Snes,Genesis,Sms,SCD'
```

Prefer a wheel tag of `macosx_14_0_arm64`. If `python -m build --wheel --no-isolation` produces a newer macOS tag, use `setup.py bdist_wheel --plat-name macosx_14_0_arm64` from the clean source copy.

After building:

```bash
$PY -m delocate.cmd.delocate_wheel --require-archs arm64 -w "$REPO/wheelhouse-post<N>-repaired" -v <macos-wheel>
$PY scripts/strip_macos_wheel.py "$REPO/wheelhouse-post<N>-repaired/<macos-wheel>"
```

Validate that the final wheel:

- Is named `stable_retro_turbo-<version>-cp314-cp314-macosx_14_0_arm64.whl`
- Contains `_retro.cpython-314-darwin.so`
- Does not contain a `cp311` extension
- Contains public cores `gambatte`, `fceumm`, `snes9x`, and `genesis_plus_gx`
- Passes import smoke tests for `stable_retro` and `stable_retro._retro`
- Passes `twine check`

The build may emit `fatal: not a git repository` metadata warnings from the temp copy and third-party compiler warnings. Report them, but they are not failures by themselves.

## Linux Manylinux Wheel

Build Linux with Docker and `cibuildwheel` from `linux-src-clean`. Docker access may require an escalated command in the sandbox.

Use:

```bash
CIBW_BUILD='cp314-manylinux_x86_64' CIBW_ARCHS_LINUX='x86_64' "$PY" -m cibuildwheel --platform linux --output-dir "$REPO/wheelhouse-post<N>-linux"
```

The expected pyproject/cibuildwheel environment includes:

```bash
CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release -DBUILD_MANYLINUX=ON -DBUILD_CORES=gb;nes;snes;genesis -DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF -DBUILD_N64=OFF'
STABLE_RETRO_PUBLIC_CORES='gambatte,fceumm,snes9x,genesis_plus_gx'
STABLE_RETRO_PUBLIC_DATA_PLATFORMS='GameBoy,Nes,Snes,Genesis,Sms,SCD'
```

Validate that the final Linux wheel:

- Is named `stable_retro_turbo-<version>-cp314-cp314-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl`
- Was repaired by `auditwheel`
- Was stripped by `scripts/strip_linux_wheel.py`
- Contains `_retro.cpython-314-x86_64-linux-gnu.so`
- Does not contain a `cp311` extension
- Contains public cores `gambatte`, `fceumm`, `snes9x`, and `genesis_plus_gx`
- Passes import smoke tests inside the cibuildwheel container

If Linux subagents hang around Docker or cibuildwheel, inspect for partial artifacts, close the stuck work, and run the Linux build directly with Docker escalation. Do not trust stale wheels in copied `dist/` directories.

## Parallelization

When subagents are available and the user asks to parallelize, split macOS and Linux builds into separate workers after the version bump and source copies are ready. Give each worker:

- The exact source copy path
- The target wheelhouse path
- The version string
- The platform-specific command and validation checklist from this skill

Keep the main agent responsible for final host-side validation and the final deploy command.

## Final Validation

Run host-side checks across both final wheelhouses:

```bash
.venv314/bin/python -m twine check wheelhouse-post<N>-repaired/*.whl wheelhouse-post<N>-linux/*.whl
shasum -a 256 wheelhouse-post<N>-repaired/*.whl wheelhouse-post<N>-linux/*.whl
```

Also inspect wheel contents with Python `zipfile` or an equivalent command to confirm:

- The version in the filename is the bumped post version
- The expected `cp314` extension exists for each platform
- No `cp311` extension exists
- All four public cores are present
- No ROM payloads such as `stable_retro/data/**/rom.nes`, `rom.sfc`, `rom.smc`, `rom.gb`, `rom.gbc`, `rom.md`, `rom.gen`, `rom.sms`, or `rom.bin` are present. `rom.sha` metadata is allowed.

## Post-Build Deploy Command

Always include the concrete `twine upload` command in the final response after a successful build. Do this even if the user only asked for builds, because the next action is usually publishing.

Use the exact wheel paths and version that were just built:

```bash
.venv314/bin/python -m twine upload \
  wheelhouse-post<N>-repaired/stable_retro_turbo-<version>-cp314-cp314-macosx_14_0_arm64.whl \
  wheelhouse-post<N>-linux/stable_retro_turbo-<version>-cp314-cp314-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl
```

If `twine` is not installed in `.venv314` or the user asks how to deploy, also include this setup command before the upload command:

```bash
.venv314/bin/python -m pip install twine
```
