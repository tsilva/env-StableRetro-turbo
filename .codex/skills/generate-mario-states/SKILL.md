---
name: generate-mario-states
description: Generate and validate Super Mario Bros NES stable-retro state files in env-StableRetro-turbo. Use when asked to create missing SuperMarioBros-Nes-v0 Level*.state files, inspect which Mario checkpoint states are missing, compare generated states to NESMaps screenshots, or deploy generated SMB states into a local installed env-stableretro-turbo package.
---

# Generate Mario States

## Workflow

Use the local source of truth first:

1. List existing files in `stable_retro/data/stable/SuperMarioBros-Nes-v0`.
2. Treat existing `.state` files as protected. Do not overwrite them unless the user explicitly asks.
3. For normal SMB levels, generate only missing `LevelW-L.state` files with `scripts/generate_smb_states.py`.
4. Render screenshots for every generated state and compare them against NESMaps before calling the state valid.
5. If the user wants the states available in another repo's installed package, copy only the generated files into that venv's `site-packages/stable_retro/data/stable/SuperMarioBros-Nes-v0/` and verify from that repo's cwd so the local checkout does not shadow the installed wheel.

## Quick Commands

Generate missing states into the repo data folder:

```bash
python3 .codex/skills/generate-mario-states/scripts/generate_smb_states.py Level2-2 Level2-3 Level2-4 --screens-dir /private/tmp/smb_generated_screens
```

Generate into a temporary folder for validation without touching repo data:

```bash
python3 .codex/skills/generate-mario-states/scripts/generate_smb_states.py Level2-2 --output-dir /private/tmp/smb_states --screens-dir /private/tmp/smb_screens
```

Verify available states from the package actually being used:

```bash
python3 -c "import stable_retro.data; print(stable_retro.data.list_states('SuperMarioBros-Nes-v0'))"
```

## Validation

Use NESMaps as the visual reference:

`https://nesmaps.com/maps/SuperMarioBrothers/SuperMarioBrothers.html`

Prefer the full map pages and full PNGs over thumbnails. For example, `World 2-2` has a short aboveground entry strip and the underwater section in a lower row; a valid `Level2-2.state` may start in the underwater section.

For generated screenshots, check:

- HUD world label matches the requested level.
- The opening geometry matches the NESMaps full map for that level.
- Area type is correct: underground, platform/bridge, water, or castle.
- Mario is controllable after reset; avoid title/demo/intermediate screens.

## SMB State Details

Read `references/smb-level-data.md` for the RAM addresses and area-pointer table used by the generator.

Important guardrails:

- Python prepends the current working directory to `sys.path`; running another repo's venv Python from this checkout can falsely import the local source tree. Verify installed-wheel behavior from the consumer repo's cwd.
- The generated gzip state should decompress to the native NES state size seen locally, currently `13321` bytes.
- `2-2` and `7-2` are water stages. Their final runtime `AreaPointer` may differ after the loader settles, so validate by HUD, area type, and screenshot rather than by final pointer alone.
