---
name: benchmark-build
description: Run and report stable-retro-turbo vector benchmarks from the current checkout or built artifacts. Use when the user asks to benchmark a stable-retro-turbo version/build, compare current throughput against vanilla post0, benchmark SuperMarioBros-NES/Mario, verify benchmark numbers after a release build, or run the repo's standardized benchmark profiles.
---

# Benchmark Build

## Overview

Use this skill to benchmark `stable-retro-turbo` with the repo's standardized vector benchmark profiles. By default, benchmark the current source checkout against vanilla `.post0`; this keeps ordinary benchmark requests aligned with the code currently under review. Measure a built wheel only when the user explicitly asks for a wheel/release benchmark or when the turn follows `$create-build` and the artifact is the object being validated.

Default Mario benchmark: `SuperMarioBros-Nes-v0`, state `Level1-1`, profile `supermario-level1-1` from `scripts/benchmark_vec_env.json`. Do not use `State.NONE` for ordinary user-facing benchmark numbers.

The benchmarker supports both the current fused native vector path and vanilla `.post0` upstream-style builds:

- Current builds: `--backend auto` resolves to `backend=native` when `StableRetroNativeVecEnv` exists.
- Vanilla `.post0`: `--backend auto` resolves to `backend=subproc` over classic `RetroEnv`, with benchmark-side preprocessing wrappers for crop, grayscale, resize, frame skip, maxpool, and frame stack.

## Default Comparison Protocol

Whenever the user asks to benchmark without naming a wheel or published version, benchmark the current checkout first. When the user names a version, wheel, or release artifact, benchmark that requested artifact first.

Then benchmark vanilla `stable-retro-turbo==1.0.0.post0` with the same SuperMarioBros-NES profile and report the comparison. The purpose is to compare the current native vector path against the original upstream-style `.post0` regular vector path.

Use this default sampling policy unless the user specifies otherwise:

- Run `SuperMarioBros-Nes-v0` / `Level1-1` with profile `supermario-level1-1`.
- Use sampled actions for the primary comparison.
- Run a short smoke for each version before timing.
- Run at least 3 full timing samples per version, using the same `--seconds`, `--warmup-steps`, env count, action mode, and backend auto-selection policy.
- Prefer 30-second timing samples for final numbers.
- Compute arithmetic mean and sample standard deviation for each version.
- Compute speedup as `requested_version_mean_steps_per_second / post0_mean_steps_per_second`.
- If fixed-action runs are useful for diagnosing noise, report them separately and do not use them for the main speedup unless the user explicitly asks.

The final answer must include a table with at least:

```text
Version/build | Backend | Samples | Mean steps/s | Std steps/s | Speedup vs post0
```

Include the individual sample values below the table or in a compact `samples=[...]` field so variance is auditable.

After every completed benchmark run, update the latest benchmark comparison
table in `README.md` with the requested/current build first and the `.post0`
baseline second. Keep the table synchronized with the final answer's samples,
mean, sample standard deviation, and speedup multiplier.

## Source Of Truth

1. Inspect the current repo state first:

```bash
git status --short --branch
rg -n "supermario-level1-1|megaman-level1" scripts/benchmark_vec_env.json scripts/benchmark_vec_env.py
```

2. Use `scripts/benchmark_vec_env.py` as the benchmark entrypoint and `scripts/benchmark_vec_env.json` as the profile source.
3. Prefer leaving `--backend auto` unless the user asks for a specific path. Force `--backend native` for fused `StableRetroNativeVecEnv`; force `--backend subproc` for the vanilla upstream comparison path; use `--backend dummy` only for single-process debugging.
4. If measuring the current checkout, verify that `stable_retro.__file__` points inside the repo checkout and that `stable_retro._retro` imports successfully:

```bash
python -c "import stable_retro, stable_retro._retro; print(stable_retro.__file__); print(stable_retro._retro.__file__)"
```

5. If measuring a wheel, verify the exact import path before timing:

```bash
python -c "import stable_retro, stable_retro._retro; import importlib.metadata as md; print(stable_retro.__file__); print(md.metadata('stable-retro-turbo')['Name']); print(md.version('stable-retro-turbo'))"
```

The path must point to the benchmark environment's `site-packages`, not the repository source tree.

## Current Checkout Benchmark Workflow

Use this workflow for ordinary benchmark requests.

1. Use the repo's Python environment, normally `.venv314`, and run from the repo root so imports resolve to this checkout:

```bash
"/Users/tsilva/repos/tsilva/stable-retro-turbo/.venv314/bin/python" -c "import stable_retro, stable_retro._retro; print(stable_retro.__file__); print(stable_retro._retro.__file__)"
```

2. Check whether the ROM is available:

```bash
"/Users/tsilva/repos/tsilva/stable-retro-turbo/.venv314/bin/python" -c "import stable_retro as retro; print(retro.data.get_romfile_path('SuperMarioBros-Nes-v0'))"
```

3. Run the standard Mario smoke and timing commands from the repo root with `.venv314/bin/python`.

## Wheel Benchmark Workflow

Use this workflow when the user asks for a wheel/release benchmark or when validating artifacts after `$create-build`.

1. Pick the exact wheel path. For macOS arm64 post releases, expect a path like:

```text
wheelhouse-post<N>-repaired/stable_retro_turbo-<version>-cp314-cp314-macosx_14_0_arm64.whl
```

2. Create a throwaway venv under `/private/tmp` and install the wheel:

```bash
BENCH_ROOT="$(mktemp -d /private/tmp/stable-retro-turbo-bench.XXXXXX)"
/Users/tsilva/repos/tsilva/stable-retro-turbo/.venv314/bin/python -m venv "$BENCH_ROOT/venv"
"$BENCH_ROOT/venv/bin/python" -m pip install /absolute/path/to/stable_retro_turbo-<version>-cp314-cp314-macosx_14_0_arm64.whl
"$BENCH_ROOT/venv/bin/python" -m pip install stable-baselines3
```

If dependency install fails because sandbox networking is blocked, rerun the same pip command with network escalation.

3. Verify the wheel import path with the command in "Source Of Truth".
4. Check whether the ROM is available in that venv:

```bash
"$BENCH_ROOT/venv/bin/python" -c "import stable_retro as retro; print(retro.data.get_romfile_path('SuperMarioBros-Nes-v0'))"
```

5. If the ROM is missing, import local ROMs from the user's ROM folder:

```bash
"$BENCH_ROOT/venv/bin/python" -m stable_retro.import /Users/tsilva/Desktop/roms
```

Do not treat a missing ROM in a clean wheel environment as a benchmark tooling failure.

## Standard Mario Timing

Run a short smoke first:

```bash
"$BENCH_ROOT/venv/bin/python" /absolute/path/to/repo/scripts/benchmark_vec_env.py \
  --profiles-json /absolute/path/to/repo/scripts/benchmark_vec_env.json \
  --profile supermario-level1-1 \
  --seconds 2 \
  --warmup-steps 8
```

Then run at least three 30-second samples. For `.post0`, the same command should print `backend=subproc`; for current builds, it should print `backend=native`.

```bash
"$BENCH_ROOT/venv/bin/python" /absolute/path/to/repo/scripts/benchmark_vec_env.py \
  --profiles-json /absolute/path/to/repo/scripts/benchmark_vec_env.json \
  --profile supermario-level1-1 \
  --seconds 30 \
  --warmup-steps 32
```

If results are noisy, run additional 10- or 30-second samples and report the spread rather than hiding it.

For a lower-variance isolation check, add a fixed-action run:

```bash
"$BENCH_ROOT/venv/bin/python" /absolute/path/to/repo/scripts/benchmark_vec_env.py \
  --profiles-json /absolute/path/to/repo/scripts/benchmark_vec_env.json \
  --profile supermario-level1-1 \
  --seconds 30 \
  --warmup-steps 32 \
  --fixed-actions
```

## Life-Loss Benchmark

When benchmarking Mario first-life-loss behavior, measure the opt-in native path with:

```python
terminate_on_life_loss=True
life_variable="lives"
```

If `scripts/benchmark_vec_env.py` does not expose those flags yet, run a small inline Python benchmark that imports the script's `_build_native_vec` helper and adds only those two kwargs. Keep all other settings aligned with `supermario-level1-1`:

```python
env_kwargs = {
    "render_mode": "rgb_array",
    "obs_resize": (84, 84),
    "obs_grayscale": True,
    "obs_crop": (32, 0, 0, 0),
    "obs_resize_algorithm": "area",
    "frame_skip": 4,
    "frame_stack": 4,
    "maxpool_last_two": True,
    "info_mode": "terminal",
    "obs_layout": "hwc",
    "terminate_on_life_loss": True,
    "life_variable": "lives",
}
```

Use the same `num_envs=32`, `num_threads=16`, warmup, duration, and action mode as the default benchmark you are comparing against.

## Reporting

Include these details in the final answer:

- Exact wheel path or source checkout measured.
- Verified `stable_retro.__file__` import path and package version.
- ROM source or confirmation that the ROM was already available.
- Benchmark profile and resolved settings: game, state, env count, thread count, resize, grayscale, crop, frame skip, frame stack, info mode, action mode.
- Individual throughput samples, not only an average, when variance is visible.
- Confirmation that `README.md` was updated with the latest comparison table.
- A concise comparison to prior remembered baseline only if memory was used and clearly labeled as memory-derived.

Useful interpretation guardrails:

- Mario sampled-action benchmark timing can vary materially by trajectory and system load.
- A clean wheel should not include ROM payloads; import the ROM into the benchmark environment.
- `State.NONE` numbers are emulator hot-path diagnostics, not training-representative results.
- `.post0` is expected to benchmark through `subproc_vec_retro`, not `native_vec_fused`, because it predates `StableRetroNativeVecEnv`.
