# Publishing

This repository publishes the `env-stableretro-turbo` distribution while
preserving the upstream-compatible `stable_retro` and `retro` imports and the
existing `stable-retro-turbo` command.

## Policy

- Publish only through `.github/workflows/release.yml` and the protected `pypi` environment.
- Keep the downstream post-release version in `stable_retro/VERSION.txt` aligned across every artifact.
- Publish binary wheels only for Apple-silicon macOS and x86-64 Linux, plus one source distribution.
- Never publish this fork as the upstream `stable-retro` distribution.

The migration release also publishes one final metadata-only
`stable-retro-turbo` wheel and source distribution that depend exactly on
`env-stableretro-turbo` at the same version. Later releases publish only the
new distribution.
