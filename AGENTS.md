# Project Notes

## Mario State Generation

Use the repo-local `generate-mario-states` skill when creating, validating, or deploying `SuperMarioBros-Nes-v0` `Level*.state` files.

Key guardrails:

- Do not overwrite existing Mario state files unless explicitly requested.
- Validate generated screenshots against NESMaps before treating a state as correct.
- When checking a consumer repo's installed package, run Python from that consumer repo's cwd so this checkout does not shadow the installed wheel.
