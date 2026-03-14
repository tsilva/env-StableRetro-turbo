<div align="center">
  <img src="logo.png" alt="stable-retro-apple-silicon" width="512"/>

  [![PyPI](https://img.shields.io/pypi/v/stable-retro-apple-silicon)](https://pypi.org/project/stable-retro-apple-silicon/)
  [![Python](https://img.shields.io/pypi/pyversions/stable-retro-apple-silicon)](https://pypi.org/project/stable-retro-apple-silicon/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

  **🎮 Native Apple Silicon wheels for retro game reinforcement learning 🍎**

</div>

## 🔍 Overview

**The Pain:** Installing [`stable-retro`](https://github.com/Farama-Foundation/stable-retro) on Apple Silicon means building from source — fighting CMake, missing dependencies, and wasted hours.

**The Solution:** Pre-built `arm64` wheels published to PyPI. One `pip install` and you're running retro game environments natively.

**The Result:** Go from zero to training in under 60 seconds on your M-series Mac.

## 🚀 Quick Start

```bash
pip install stable-retro-apple-silicon
```

```python
import stable_retro as retro

env = retro.make("Alleyway-GameBoy-v0", render_mode="rgb_array")
```

The deprecated compatibility alias still works:

```python
import retro
```

## ✨ What You Get

- Native Apple Silicon wheels published to PyPI
- Matching downloadable wheel files attached to GitHub Releases
- A maintained Apple Silicon packaging layer on top of upstream [`stable-retro`](https://github.com/Farama-Foundation/stable-retro)
- The same `stable_retro` and `retro` import surface users expect

## 📦 Release Matrix

| Item | Value |
| --- | --- |
| Package | `stable-retro-apple-silicon` |
| CPU | Apple Silicon `arm64` |
| macOS | `14.0+` |
| Python | `3.9` to `3.12` |
| Wheel contents | code, public cores, game metadata |
| Public cores | Game Boy, NES, SNES, Sega Master System |

## 🔧 Build Notes

- Apple Silicon wheel builds use a restricted public core set: `gambatte`, `fceumm`, `snes9x`, `genesis_plus_gx`
- CapnProto is disabled in the public wheel build path
- Release automation publishes wheels to PyPI and also attaches them to GitHub Releases

## 📚 Upstream Docs

The underlying project is `stable-retro`, so the upstream documentation remains relevant for APIs and integration metadata:

- [Supported emulators](docs/supported_emulators.md)
- [Supported games/envs](docs/supported_games.md)
- [macOS installation notes](docs/macos_installation.md)
- [Linux installation notes](docs/linux_installation.md)

## 📋 Publishing

See [`PUBLISHING.md`](PUBLISHING.md) for the release checklist and local wheel verification commands.
