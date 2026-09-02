# Cross-provider parity

Original `stable-retro==1.0.1` is the semantic authority for the Turbo fork.
The fork must match that authority directly; comparing only against another
Turbo environment can reproduce the same bug on both sides.

TurboBench's immutable profiles exercise two reference integrations:

- `supermario/world1-v1` assigns World 1 levels 1-1 through 1-4 across
  lanes and runs shapes 1 and 4 for 4,096 seeded transitions.
- `breakout/start-v1` uses the cartridge `Start` state and runs shapes 1 and 4
  for 4,096 seeded transitions.

TurboBench compares the native scalar authority with `RetroVecEnv` across
processed observations, lossless raw frames, actions, rewards, termination and
truncation, selected info, lane resets, RAM, and snapshot continuation. Mario
uses the canonical 2 KiB NES CPU RAM address space. The only allowed Mario
frame representation conversion removes the copied low bits from Stable
Retro's RGB888 expansion to recover the exact RGB565 native pixel code.
Breakout public RGB bytes are compared directly.

With lawful local ROMs imported, run:

```bash
turbobench parity supermario/world1-v1 \
  --candidate env-stableretro-turbo@checkout:"$PWD" \
  --allow-dirty --quick

turbobench parity breakout/start-v1 \
  --candidate env-stableretro-turbo@checkout:"$PWD" \
  --allow-dirty --quick
```

For release certification, run both profiles against the exact final wheel and
verify the receipts:

```bash
turbobench verify-parity /external/evidence/mario-env-stableretro-turbo \
  --require-canonical --require-provider env-stableretro-turbo
turbobench verify-parity /external/evidence/breakout-env-stableretro-turbo \
  --require-canonical --require-provider env-stableretro-turbo
```

The canonical gate rejects shortened workloads, missing shapes, every checkout
candidate, dirty overrides, an authority override, the wrong Stable Retro
version, failed exact checks, and receipts for a different candidate.
`--allow-dirty`, `--steps`, or `--shapes` remain useful for diagnosis but cannot
produce release evidence.
