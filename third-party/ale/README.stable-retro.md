This directory vendors the native Arcade Learning Environment sources from
`ale-py` v0.12.0 (upstream commit
`94c24368664b8539c53857522e50652ddcc44b20`).  Stable Retro owns and builds
this copy into `_retro` so Atari vector lifecycle fixes do not depend on an
unreleased or monkeypatched `ale-py` wheel.

Local changes are intentionally limited to the vector backend and build glue:

- native `Disabled` autoreset support;
- lane-local masked reset scheduling and terminal-lane step validation;
- dependency-free area resizing (equivalent coverage-weighted semantics to
  OpenCV `INTER_AREA` for downscaling);
- integration into Stable Retro's existing pybind11 extension.

The vendored code remains licensed under GPL-2.0-only; see `LICENSE.md`.
