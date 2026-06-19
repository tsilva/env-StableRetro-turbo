# Publishing

This repository keeps the upstream Stable Retro package identity. The repository
may be named `stable-retro-turbo`, but the Python package metadata remains
`stable-retro`.

## Policy

- Do not publish fork builds to PyPI under a separate package name.
- Do not publish fork builds to PyPI as `stable-retro`.
- Use GitHub Releases for branch-specific wheel artifacts when needed.
- Keep `stable_retro/VERSION.txt` aligned with upstream unless preparing a
  deliberate downstream release strategy.

## Release Artifacts

The release workflow builds macOS Apple Silicon and Linux x86_64 wheels and
attaches them to GitHub Releases. Those artifacts are for validation and direct
download, not PyPI publication.
