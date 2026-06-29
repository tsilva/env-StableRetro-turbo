---
name: build
description: Build and validate the next stable-retro-turbo post-version release wheels. Use when the user says /build, asks to do a new build with the next version, or asks for macOS and Linux stable_retro_turbo wheels for publishing with twine.
---

# Build

Use this skill to create publish-ready `stable_retro_turbo` wheels for the next
`.postN` version on macOS arm64 and Linux x86_64. Keep release mechanics in
`scripts/release_build.py`; do not retype the long shell workflow unless the
script is genuinely blocked.

## Flow

1. Inspect the worktree and preserve unrelated user changes:

```bash
git status --short --branch
```

2. Confirm the distribution name:

```bash
.venv314/bin/python setup.py --name
```

It must print `stable-retro-turbo`.

3. Bump to the next post version:

```bash
.venv314/bin/python scripts/release_build.py bump-version --write
```

4. If current-turn source changes have not been regression-tested, run focused
tests before building:

```bash
MPLCONFIGDIR=/tmp/matplotlib-stable-retro PYTHONPATH=. .venv314/bin/python -m pytest tests/test_python/test_vec_env.py -q
```

When emulator behavior changed, also run from `tests/`:

```bash
./test-emulator --gtest_filter='EmulatorCore/EmulatorTest.*/*Nes'
```

5. Create clean source copies under `/private/tmp`:

```bash
.venv314/bin/python scripts/release_build.py prepare-sources
```

Use the JSON output paths for the platform builds. The script excludes stale
build outputs, compiled artifacts, wheelhouses, local venv state, CMake cache
files, and actual ROM payloads, then verifies the Linux copy is clean.

6. Print the exact platform build commands:

```bash
.venv314/bin/python scripts/release_build.py build-commands \
  --macos-src /private/tmp/<build-root>/macos-src \
  --linux-src /private/tmp/<build-root>/linux-src-clean
```

Run the macOS commands locally. Run the Linux `cibuildwheel` command from the
clean Linux source copy; Docker access may require escalation in the sandbox.

7. After the macOS wheel is repaired and stripped, smoke-test it from outside
the checkout:

```bash
.venv314/bin/python scripts/release_build.py smoke-macos-wheel \
  wheelhouse-post<N>-repaired/stable_retro_turbo-<version>-cp314-cp314-macosx_14_0_arm64.whl
```

The Linux wheel import smoke is run by `cibuildwheel` inside the container.

8. Run final host-side validation:

```bash
.venv314/bin/python scripts/release_build.py final-check
```

This audits both wheel contents, runs `twine check`, prints SHA256 hashes, and
prints the exact `twine upload` command.

## Final Response

Report the two wheel paths, SHA256 hashes, validation results, notable warnings,
and the concrete upload command printed by `final-check`.
