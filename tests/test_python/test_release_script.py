import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _release_module():
    path = ROOT / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_promotes_unreleased_changelog(tmp_path, monkeypatch):
    release = _release_module()
    changes = tmp_path / "CHANGES.md"
    changes.write_text(
        "# Changelog\n\n## Unreleased\n\n* New behavior.\n\n## 1.0.0\n\n* Old behavior.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "CHANGES", changes)

    release.promote_changelog("1.0.1.post1", release_date="2026-07-20")

    assert changes.read_text(encoding="utf-8") == (
        "# Changelog\n\n## Unreleased\n\n* Nothing yet.\n\n"
        "## 1.0.1.post1 - 2026-07-20\n\n* New behavior.\n\n"
        "## 1.0.0\n\n* Old behavior.\n"
    )


def test_release_promotes_and_stages_changelog_before_tag_and_push():
    source = (ROOT / "scripts" / "release.py").read_text(encoding="utf-8")
    main = source[source.index("def main()") :]

    assert main.index("promote_changelog(version)") < main.index(
        "tag = create_commit_and_tag(version)"
    )
    assert main.index("tag = create_commit_and_tag(version)") < main.index(
        "push_release(remote, branch, tag, args.dry_run_push)"
    )
    assert (
        '"CHANGES.md"'
        in source[
            source.index("def create_commit_and_tag") : source.index("def push_release")
        ]
    )
