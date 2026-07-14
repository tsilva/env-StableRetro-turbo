# Project Notes

## Product Specifications

Before any task in this repository, use the `$specs-author` skill to read the root `SPECS.md`. Use `$specs-author` whenever reading or writing `SPECS.md`.

## Mario State Generation

Use the repo-local `generate-mario-states` skill when creating, validating, or deploying `SuperMarioBros-Nes-v0` `Level*.state` files.

Key guardrails:

- Do not overwrite existing Mario state files unless explicitly requested.
- Validate generated screenshots against NESMaps before treating a state as correct.
- When checking a consumer repo's installed package, run Python from that consumer repo's cwd so this checkout does not shadow the installed wheel.

## Beast Benchmarking

Run performance benchmarks on `beast-2` or `beast-3`, not Modal. Resolve and inspect both hosts through the sibling `rlab` machine registry and its machine-scoped commands (`rlab fleet ps --machine ...` and `rlab fleet watch --machine ... --once --no-color --no-tui`), then choose the freer reachable machine. Use the selected host's `rlab`-pinned runtime image or environment so the benchmark matches training. Do not stop, throttle, or otherwise disturb active training containers; if no host is idle, keep probes short and low-priority, label their absolute timing as contention-affected, and use live split throughput metrics as the authoritative full-loop evidence.
