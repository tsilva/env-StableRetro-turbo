# Publishing

This repository publishes the `env-stableretro-turbo` distribution with the
`env_stableretro_turbo` import package, the upstream-compatible `retro` shim,
and the `env-stableretro-turbo` command.

## Policy

- Publish only through `.github/workflows/release.yml` and the protected `pypi` environment.
- Keep the downstream post-release version in `env_stableretro_turbo/VERSION.txt` aligned across every artifact.
- Publish binary wheels only for Apple-silicon macOS and x86-64 Linux, plus one source distribution.
- Never publish this fork as the upstream `stable-retro` distribution.
