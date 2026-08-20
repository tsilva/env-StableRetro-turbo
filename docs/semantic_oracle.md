# Stable Retro semantic oracle

Original `stable-retro==1.0.1` is the semantic authority for the Turbo fork.
The fork must match that authority directly; comparing only against another
Turbo environment can reproduce the same bug on both sides.

TurboBench's immutable v2 profiles exercise two reference integrations:

- `supermario/canonical-v2` assigns World 1 levels 1-1 through 1-4 across
  lanes and runs shapes 1 and 4 for 4,096 seeded transitions.
- `breakout/start-v2` uses the cartridge `Start` state and runs shapes 1 and 4
  for 4,096 seeded transitions.

The oracle compares the native scalar authority with `RetroVecEnv` across
processed observations, lossless raw frames, actions, rewards, termination and
truncation, selected info, lane resets, RAM, and snapshot continuation. Mario
uses the canonical 2 KiB NES CPU RAM address space. The only allowed Mario
frame representation conversion removes the copied low bits from Stable
Retro's RGB888 expansion to recover the exact RGB565 native pixel code.
Breakout public RGB bytes are compared directly.

With lawful local ROMs imported, run:

```bash
turbobench oracle supermario/canonical-v2 \
  --left stable-retro@1.0.1 \
  --right env-stableretro-turbo@checkout:"$PWD" \
  --output /external/evidence/mario-stable-retro-turbo

turbobench oracle breakout/start-v2 \
  --left stable-retro@1.0.1 \
  --right env-stableretro-turbo@checkout:"$PWD" \
  --output /external/evidence/breakout-stable-retro-turbo
```

After publishing the candidate, regenerate both commands with
`env-stableretro-turbo@VERSION` instead of the checkout selector, then verify the
published-release receipts:

```bash
turbobench verify-oracle /external/evidence/mario-stable-retro-turbo \
  --require-canonical --require-provider env-stableretro-turbo
turbobench verify-oracle /external/evidence/breakout-stable-retro-turbo \
  --require-canonical --require-provider env-stableretro-turbo
```

The canonical gate rejects shortened workloads, missing shapes, every checkout
candidate, dirty overrides, a non-PyPI authority, the wrong Stable Retro
version, failed exact checks, and receipts for a different candidate.
`--allow-dirty`, `--steps`, or `--shapes` remain useful for diagnosis but cannot
produce release evidence.
