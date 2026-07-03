---
name: build-release
description: Build, validate, and prepare upload for the next stable-retro-turbo post-version release wheels. Use when the user says /build-release, asks to build a release with the next version, asks to publish with the next version, or asks for macOS and Linux stable_retro_turbo wheels for publishing with twine.
---

# Build Release

Use this skill to create publish-ready `stable_retro_turbo` wheels for the next
`.postN` version on macOS arm64 and Linux x86_64. Keep release mechanics in
`scripts/release_build.py`; do not retype the long shell workflow unless the
script is genuinely blocked.

Do not run this skill unless the current branch is fully clean and synchronized:
all changes committed, no untracked files, upstream configured, remote state
fetched, and no commits ahead of or behind upstream. Stop before bumping the
version or running any release-build command if any part of this gate fails. Do
not commit, push, pull, or clean files to satisfy this gate unless the user
explicitly asks for that.

## Flow

1. Verify the release gate:

```bash
git status --short --branch
```

The output must contain only the branch line. If there are modified, deleted, or
untracked files, stop and tell the user the tree must be committed or cleaned
before publishing.

```bash
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

This must print an upstream branch. If it fails, stop and ask the user to set an
upstream before publishing.

```bash
git fetch --prune
git rev-list --left-right --count HEAD...@{u}
```

The rev-list output must be two zeroes, `0 0` (Git usually separates them with a
tab). If the first number is nonzero, local commits have not been pushed; stop.
If the second number is nonzero, the branch has not been pulled; stop. If both
are nonzero, the branch has diverged; stop.

2. Confirm the distribution name:

```bash
.venv314/bin/python setup.py --name
```

It must print `stable-retro-turbo`.

3. Verify the tracked upstream Stable Retro base version:

```bash
git fetch --prune upstream main
UPSTREAM_STABLE_RETRO_VERSION="$(git show upstream/main:stable_retro/VERSION.txt | tr -d '[:space:]')"
CURRENT_TURBO_VERSION="$(tr -d '[:space:]' < stable_retro/VERSION.txt)"
CURRENT_TURBO_BASE="${CURRENT_TURBO_VERSION%%.post*}"
printf 'upstream stable-retro: %s\ncurrent stable-retro-turbo: %s\n' \
  "${UPSTREAM_STABLE_RETRO_VERSION}" \
  "${CURRENT_TURBO_VERSION}"
```

The `stable-retro-turbo` version must always be prefixed by the upstream Stable
Retro version currently tracked on `upstream/main`. If
`CURRENT_TURBO_BASE != UPSTREAM_STABLE_RETRO_VERSION`, start a new post series
on the tracked upstream base and use:

```bash
TARGET_VERSION="${UPSTREAM_STABLE_RETRO_VERSION}.post1"
```

For example, if upstream is `1.0.1` and the current turbo version is
`1.0.0.post23`, the next release target is `1.0.1.post1`, not
`1.0.0.post24`. Stop and ask the user before publishing if the tracked upstream
base is surprising, cannot be read, or does not match the intended release
line.

4. Find the next post version and tag the clean current commit:

```bash
if [ -z "${TARGET_VERSION:-}" ]; then
  TARGET_VERSION="$(.venv314/bin/python scripts/release_build.py bump-version)"
fi
RELEASE_TAG="v${TARGET_VERSION}"
if git rev-parse --verify --quiet "refs/tags/${RELEASE_TAG}"; then
  echo "Tag already exists: ${RELEASE_TAG}"
  exit 1
fi
```

`TARGET_VERSION` is the package version, and `RELEASE_TAG` is the Git release
tag. Existing tags in this repo use the `v` prefix. If the tag already exists,
stop; do not overwrite or move it.

```bash
git tag "${RELEASE_TAG}" HEAD
git rev-parse "${RELEASE_TAG}^{commit}"
git rev-parse HEAD
```

The two commit hashes must match. This tags the already-clean, already-pushed,
already-pulled current commit before the version bump dirties the working tree.

5. Bump to the tagged post version:

```bash
.venv314/bin/python scripts/release_build.py bump-version --to "${TARGET_VERSION}" --write
```

6. If current-turn source changes have not been regression-tested, run focused
tests before building:

```bash
MPLCONFIGDIR=/tmp/matplotlib-stable-retro PYTHONPATH=. .venv314/bin/python -m pytest tests/test_python/test_vec_env.py -q
```

When emulator behavior changed, also run from `tests/`:

```bash
./test-emulator --gtest_filter='EmulatorCore/EmulatorTest.*/*Nes'
```

7. Create clean source copies under `/private/tmp`:

```bash
.venv314/bin/python scripts/release_build.py prepare-sources
```

Use the JSON output paths for the platform builds. The script excludes stale
build outputs, compiled artifacts, wheelhouses, local venv state, CMake cache
files, and actual ROM payloads, then verifies the Linux copy is clean.

8. Print the exact platform build commands:

```bash
.venv314/bin/python scripts/release_build.py build-commands \
  --macos-src /private/tmp/<build-root>/macos-src \
  --linux-src /private/tmp/<build-root>/linux-src-clean
```

Run the macOS commands locally. Run the Linux `cibuildwheel` command from the
clean Linux source copy; Docker access may require escalation in the sandbox.

9. After the macOS wheel is repaired and stripped, smoke-test it from outside
the checkout:

```bash
.venv314/bin/python scripts/release_build.py smoke-macos-wheel \
  wheelhouse-post<N>-repaired/stable_retro_turbo-<version>-cp314-cp314-macosx_14_0_arm64.whl
```

The Linux wheel import smoke is run by `cibuildwheel` inside the container.

10. Run final host-side validation:

```bash
.venv314/bin/python scripts/release_build.py final-check
```

This audits both wheel contents, runs `twine check`, prints SHA256 hashes, and
prints the exact `twine upload` command.

## Final Response

Report the release tag, the two wheel paths, SHA256 hashes, validation results,
notable warnings, and the concrete upload command printed by `final-check`.
