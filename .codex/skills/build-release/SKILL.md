---
name: build-release
description: Launch and monitor the env-stableretro-turbo release flow. Use when the user says /build-release, asks to build/publish/cut the next release, asks for a specific post-version release, or wants the PyPI link after publishing.
---

# Build Release

Use this skill to cut an `env-stableretro-turbo` release by delegating to the
repo-owned release target. Do not recreate the old hand-written build workflow
unless `make release` itself is blocked and the user explicitly asks for manual
recovery.

`make release` owns the release mechanics:

- creates/updates `.venv314`;
- verifies the tree is clean and the branch is synced with its upstream;
- checks the upstream Stable Retro base version;
- checks the target version is unused on PyPI;
- bumps `stable_retro/VERSION.txt`;
- requires non-empty, human-authored notes in the checked-in `CHANGES.md`
  `Unreleased` section;
- promotes those notes to the target version and release date, then creates a
  fresh `Unreleased` section;
- commits `CHANGES.md` with the version bump as `Release v<version>`;
- tags `v<version>`;
- atomically pushes the branch and tag;
- lets GitHub Actions build, validate, and publish the wheels through trusted
  publishing.

The release targets are exactly `macos-arm64` and `linux-x86_64`, plus a source
distribution.

## Flow

1. Do a small orientation check, but do not duplicate the release script:

```bash
git status --short --branch
sed -n '1,80p' GNUmakefile
sed -n '1,180p' scripts/release.py
```

If the tree is dirty, stop before launching and report the dirty files. Do not
commit, clean, pull, push, or switch branches unless the user asks.

2. Launch the release from the repo root:

```bash
make release
```

For an explicit target version:

```bash
RELEASE_ARGS="--to 1.0.1.postN" make release
```

Use the exact command name `make release`. If a user mistypes it as
`make releaes`, treat that as a typo unless the repo actually has a `releaes`
target.

3. Monitor the `make release` process.

If it fails before pushing, report the failing gate or command and stop. Typical
failures are a dirty tree, unsynced branch, existing tag, upstream base mismatch,
an existing PyPI version, or missing/empty release notes. Never synthesize
release notes from commits; the script may only promote human-authored
`Unreleased` prose.

If it succeeds, capture the printed tag from output such as:

```text
Released v1.0.1.postN: pushed <branch> and tag to <remote>.
GitHub Actions will build, validate, and publish the release wheels from the pushed tag.
```

4. Monitor GitHub Actions for the pushed tag. Prefer `gh` if available:

```bash
gh run list --workflow Release --branch v1.0.1.postN --limit 5
gh run watch <run-id> --exit-status
```

If `gh run list --branch <tag>` is unreliable for tag refs, use:

```bash
gh run list --workflow Release --limit 20
gh run view <run-id> --log-failed
```

Use the workflow URL as supporting context, not as the final success signal:

```text
https://github.com/tsilva/env-StableRetro-turbo/actions/workflows/release.yml
```

5. Poll PyPI until the released version exists. The release is not done until
PyPI reports files for the exact version:

```bash
.venv314/bin/python - 1.0.1.postN <<'PY'
import json
import sys
import time
import urllib.request

package = "env-stableretro-turbo"
version = sys.argv[1]
url = f"https://pypi.org/pypi/{package}/json"
for attempt in range(90):
    with urllib.request.urlopen(url, timeout=20) as response:
        releases = json.loads(response.read().decode("utf-8")).get("releases", {})
    files = releases.get(version) or []
    if files:
        print(f"https://pypi.org/project/{package}/{version}/")
        break
    print(f"waiting for {package}=={version} on PyPI ({attempt + 1}/90)")
    time.sleep(20)
else:
    raise SystemExit(f"{package}=={version} did not appear on PyPI in time")
PY
```

## Final Response

When PyPI has the version, respond with the release tag and the PyPI version
link first:

```text
Released v1.0.1.postN and it is live on PyPI:
https://pypi.org/project/env-stableretro-turbo/1.0.1.postN/
```

Also mention any relevant GitHub Actions result or failure. Do not report a
release as complete just because `make release` pushed the tag; wait for PyPI.
