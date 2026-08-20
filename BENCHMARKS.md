# Benchmarks

This page records correctness-gated, host-specific throughput comparisons for
Stable Retro Turbo. Results are not generalized across games, emulator cores,
or workloads beyond the exact profile named with each table.

## Official Super Mario Bros comparison

On 2026-08-11, `env-stableretro-turbo==1.0.1.post37` was compared directly with
original `stable-retro==1.0.1` on `beast-3` using TurboBench's immutable
`supermario/canonical-v1` profile. The comparison passed every validity gate and
produced an official, independently verified result bundle.

| Envs | Stable Retro Turbo median SPS | Stable Retro median SPS | Paired speedup | 95% paired bootstrap CI |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,417.4 | 631.9 | 2.2465x | 2.2169x–2.2572x |
| 16 | 7,066.0 | 3,321.7 | 2.1248x | 2.1037x–2.1465x |
| 32 | 8,223.0 | 3,687.4 | 2.2317x | 2.2252x–2.2370x |

SPS means environment steps per second. Values are never aggregated across
environment counts.

## Evidence

The self-verifying result bundle contains 119 hash-bound artifacts and passed
`turbobench verify` without errors.

| Field | Value |
| --- | --- |
| Bundle ID | `60219a4a4e15d0b9ad978fdcf58d40e99da84eef3e3e43b3a1003ac984f39635` |
| TurboBench | [`turbobench-cli==1.0.0`](https://pypi.org/project/turbobench-cli/1.0.0/), source commit `d986efa72c81a7d0b5ea689ac37898d8fc38732f` |
| Harness source SHA-256 | `2c64aefe52d5db7f2887b0f9d9d32c23c49f6590319a02eee5b6e2398b710319` |
| Profile SHA-256 | `326c6d47c4cc0bc2bbafdf003a430ea80cc27877f8a4144dfbb65dbea6bb2cd7` |
| Canonical ROM SHA-256 | `f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de` |

Exact correctness passed at shapes 1, 16, and 32 for policy observations,
normalized raw RGB frames, rewards, termination and truncation, selective reset
points, completion, and semantic infos. The providers ran in separate
content-addressed Python environments with the same Python minor and harness
dependencies. No diagnostic overrides were used.

## Protocol

The matched workload used `SuperMarioBros-Nes-v0`, states `Level1-1` through
`Level1-4`, frame skip 4, no max-pooling, four grayscale frames, a zeroed
32-row HUD, area resize to `84x84`, CHW output, and deterministic precomputed
actions.

Each shape used one unmeasured warmup pair followed by seven alternating AB/BA
measured pairs. Every invocation contained three repetitions of 250 vector
steps; invocation medians formed paired ratios, and a deterministic
20,000-resample paired bootstrap produced the 95% interval.

Timed SPS included preprocessing, IPC, infos, terminal detection, and selective
resets. It excluded construction, initial reset, action generation, warmup,
correctness replay, rendering, and encoding. Provider resolution and
installation completed before the correctness and timing subprocesses entered
the offline boundary.

## Reproduce

Install the exact released CLI, make the canonical ROM and state catalog
available in a Stable Retro-compatible integration under `RETRO_DATA_PATH` or
`TURBOBENCH_ASSET_ROOT`, then run:

```bash
uv tool install \
  --exclude-newer-package turbobench-cli=2026-08-12T00:00:00Z \
  turbobench-cli==1.0.0
turbobench compare supermario/canonical-v1 \
  --left env-stableretro-turbo@1.0.1.post37 \
  --right stable-retro@1.0.1
```

## Host

### `beast-3`

| Component | Specification |
| --- | --- |
| CPU | AMD Ryzen 5 7600X (Zen 4), 6 physical cores / 12 threads, boost enabled, 5.457 GHz reported maximum |
| CPU cache | 384 KiB L1, 6 MiB L2, 32 MiB L3 |
| Memory | 65,396,760,576 bytes reported (60.9 GiB) |
| OS | Ubuntu 26.04 LTS, Linux 7.0.0-29-generic, x86_64 |
| Runtime | CPython 3.14.6; `numpy==2.4.2`; `gymnasium==1.2.2` |

The one-minute system load initially exceeded TurboBench's 6.0 threshold. The
harness waited until it fell below the threshold before timing began; the gate
then passed without an override.
