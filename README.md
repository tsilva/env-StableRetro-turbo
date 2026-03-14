[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

# stable-retro-apple-silicon

This repository is a publishable Apple Silicon fork of
[stable-retro](https://github.com/Farama-Foundation/stable-retro). It keeps the
upstream source tree at the repo root, adds Apple Silicon build fixes, and
publishes ROM-safe public wheels under the `stable-retro-apple-silicon` package
name.

## Install

```bash
pip install stable-retro-apple-silicon
python -m stable_retro.import /path/to/your/roms
```

The import names stay the same:

```python
import stable_retro as retro
env = retro.make("Alleyway-GameBoy-v0", render_mode="rgb_array")
```

The deprecated compatibility alias still works:

```python
import retro
```

## Release Matrix

| Item | Value |
| --- | --- |
| Package | `stable-retro-apple-silicon` |
| CPU | Apple Silicon `arm64` |
| macOS | `14.0+` |
| Python | `3.9` to `3.12` |
| Wheel contents | code, public cores, `rom.sha` metadata, no ROM payloads |
| Public cores | Game Boy, NES, SNES, Sega Master System |

## ROM Policy

Public wheels do not bundle copyrighted ROM files. The package includes game
metadata and `rom.sha` hashes, and users import their own ROMs after install.

Local imported `rom.*` payloads are ignored by Git and excluded from build
artifacts. Only `rom.sha` files are kept.

## Build Notes

- Apple Silicon wheel builds use a restricted public core set:
  `gambatte`, `fceumm`, `snes9x`, `genesis_plus_gx`
- CapnProto is disabled in the public wheel build path
- release automation publishes wheels to PyPI and also attaches them to GitHub
  Releases

## Upstream Docs

The underlying project is still `stable-retro`, so the upstream documentation
remains relevant for APIs and integration metadata:

- [Supported emulators](docs/supported_emulators.md)
- [Supported games/envs](docs/supported_games.md)
- [macOS installation notes](docs/macos_installation.md)
- [Linux installation notes](docs/linux_installation.md)

## Publishing

See [`PUBLISHING.md`](PUBLISHING.md) for the release checklist and local wheel
verification commands.
