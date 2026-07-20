#!/usr/bin/env python3
"""Bump, commit, tag, and push a stable-retro-turbo release."""

from __future__ import annotations

import argparse
from datetime import date
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = REPO_ROOT / "scripts" / "release_build.py"
PYTHON = REPO_ROOT / ".venv314" / "bin" / "python"
VERSION_PATH = REPO_ROOT / "stable_retro" / "VERSION.txt"
CHANGES = REPO_ROOT / "CHANGES.md"
VERSION_RE = re.compile(r"^(?P<base>\d+\.\d+\.\d+)(?:\.post(?P<post>\d+))?$")


def run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    return subprocess.run(args, cwd=REPO_ROOT, env=env, check=True, text=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def ensure_clean() -> None:
    status = capture(["git", "status", "--short"])
    if status:
        raise SystemExit(f"release tree must be clean before bumping:\n{status}")


def upstream_ref() -> str:
    try:
        return capture(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit("current branch must have an upstream before cutting a release") from exc


def ensure_synced() -> tuple[str, str]:
    upstream = upstream_ref()
    if "/" not in upstream:
        raise SystemExit(f"unexpected upstream ref: {upstream}")
    remote, branch = upstream.split("/", 1)
    run(["git", "fetch", "--prune", remote])
    left_right = capture(["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    ahead, behind = [int(part) for part in left_right.split()]
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synced with {upstream} before release; "
            f"ahead={ahead} behind={behind}"
        )
    return remote, branch


def parse_version(version: str) -> tuple[str, int]:
    match = VERSION_RE.match(version)
    if match is None:
        raise SystemExit(f"unsupported version format: {version!r}")
    return match.group("base"), int(match.group("post") or 0)


def read_version() -> str:
    return VERSION_PATH.read_text(encoding="utf-8").strip()


def upstream_stable_retro_version() -> str:
    run(["git", "fetch", "--prune", "upstream", "main"])
    try:
        return capture(["git", "show", "upstream/main:stable_retro/VERSION.txt"]).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit("could not read upstream/main:stable_retro/VERSION.txt") from exc


def next_post_version(current: str, upstream_base: str) -> str:
    current_base, current_post = parse_version(current)
    parse_version(upstream_base)
    if current_base != upstream_base:
        return f"{upstream_base}.post1"
    return f"{current_base}.post{current_post + 1}"


def helper(*args: str) -> None:
    run([str(PYTHON), str(RELEASE_HELPER), *args])


def target_version(args: argparse.Namespace) -> str:
    current = read_version()
    upstream_base = upstream_stable_retro_version()
    if args.to:
        version = args.to
    else:
        version = next_post_version(current, upstream_base)
    target_base, _ = parse_version(version)
    if target_base != upstream_base and not args.allow_upstream_base_mismatch:
        raise SystemExit(
            f"target version {version} is based on {target_base}, but upstream/main is {upstream_base}; "
            "pass --allow-upstream-base-mismatch to override"
        )
    helper("check-version")
    helper("check-pypi", "--version", version)
    return version


def promote_changelog(version: str, *, release_date: str | None = None) -> None:
    text = CHANGES.read_text(encoding="utf-8")
    prefix = "# Changelog\n\n## Unreleased\n\n"
    if not text.startswith(prefix):
        raise SystemExit("CHANGES.md must begin with an Unreleased section")
    tail = text[len(prefix) :]
    separator = tail.find("\n## ")
    if separator < 0:
        unreleased = tail.strip()
        history = ""
    else:
        unreleased = tail[:separator].strip()
        history = tail[separator + 1 :].strip()
    if not unreleased or unreleased in {"* Nothing yet.", "- Nothing yet."}:
        raise SystemExit("CHANGES.md Unreleased section must describe the release")
    if re.search(rf"^## {re.escape(version)}(?:\s|$)", text, re.MULTILINE):
        raise SystemExit(f"CHANGES.md already contains release {version}")
    released = release_date or date.today().isoformat()
    updated = f"{prefix}* Nothing yet.\n\n## {version} - {released}\n\n{unreleased}\n"
    if history:
        updated += f"\n{history}\n"
    CHANGES.write_text(updated, encoding="utf-8")


def run_checks(version: str, skip_checks: bool) -> None:
    if skip_checks:
        return
    helper("check-version", "--version", version)
    run([str(PYTHON), "-m", "compileall", "-q", "scripts/release.py", "scripts/release_build.py"])


def create_commit_and_tag(version: str) -> str:
    tag = f"v{version}"
    if subprocess.run(["git", "rev-parse", "--verify", "--quiet", tag], cwd=REPO_ROOT).returncode == 0:
        raise SystemExit(f"tag already exists locally: {tag}")
    run(["git", "add", "stable_retro/VERSION.txt", "CHANGES.md"])
    run(["git", "commit", "-m", f"Release {tag}"])
    run(["git", "tag", tag, "HEAD"])
    return tag


def push_release(remote: str, branch: str, tag: str, dry_run: bool) -> None:
    args = ["git", "push", "--atomic", remote, f"HEAD:{branch}", tag]
    if dry_run:
        args.insert(2, "--dry-run")
    run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--to", help="Exact release version, for example 1.0.1.post2")
    parser.add_argument("--skip-checks", action="store_true", help="Skip local helper/compile checks")
    parser.add_argument("--dry-run-push", action="store_true", help="Create the commit and tag, but dry-run the push")
    parser.add_argument(
        "--allow-upstream-base-mismatch",
        action="store_true",
        help="Allow a target version whose base does not match upstream/main",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if not PYTHON.exists():
        raise SystemExit("expected release environment at .venv314/bin/python; run `make release`")
    ensure_clean()
    remote, branch = ensure_synced()
    version = target_version(args)
    helper("bump-version", "--to", version, "--write")
    promote_changelog(version)
    run_checks(version, args.skip_checks)
    tag = create_commit_and_tag(version)
    push_release(remote, branch, tag, args.dry_run_push)
    print()
    print(f"Released {tag}: pushed {branch} and tag to {remote}.")
    print("GitHub Actions will build, validate, and publish the release wheels from the pushed tag.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
