---
name: benchmark-build
description: Run and report stable-retro-turbo vector benchmarks from the current checkout or built artifacts. Use when the user asks to benchmark a stable-retro-turbo version/build, compare current throughput against vanilla post0, benchmark SuperMarioBros-NES/Mario, verify benchmark numbers after a release build, or run the repo's standardized benchmark profiles.
---

# Benchmark Build

## Overview

Use this skill to benchmark `stable-retro-turbo` with the repo's standardized vector benchmark profiles. By default, benchmark the current source checkout against vanilla `.post0`; this keeps ordinary benchmark requests aligned with the code currently under review. Measure a built wheel only when the user explicitly asks for a wheel/release benchmark or when the turn follows `$build` and the artifact is the object being validated.

Benchmark decisions must consider both isolated environment throughput and full SB3 PPO training-loop throughput. A candidate optimization is not considered durable unless it improves the plain vector env benchmark and the end-to-end training benchmark, because final RL experiment throughput includes rollout collection, tensor conversion, policy inference, PPO updates, logging, and synchronization overhead.

Default Mario benchmark: `SuperMarioBros-Nes-v0`, state `Level1-1`, profile `supermario-level1-1` from `scripts/benchmark_vec_env.json`. Do not use `State.NONE` for ordinary user-facing benchmark numbers.

The benchmarker supports both the current fused native vector path and vanilla `.post0` upstream-style builds:

- Current builds: `--backend auto` resolves to `backend=native` when `StableRetroNativeVecEnv` exists.
- Vanilla `.post0`: `--backend auto` resolves to `backend=subproc` over classic `RetroEnv`, with benchmark-side preprocessing wrappers for crop, grayscale, resize, frame skip, maxpool, and frame stack.

## Default Comparison Protocol

Whenever the user asks to benchmark without naming a wheel or published version, benchmark the current checkout first. When the user names a version, wheel, or release artifact, benchmark that requested artifact first.

Then benchmark vanilla `stable-retro-turbo==1.0.0.post0` with the same SuperMarioBros-NES profile and report the comparison. The purpose is to compare the current native vector path against the original upstream-style `.post0` regular vector path.

Use this default sampling policy unless the user specifies otherwise:

- Run `SuperMarioBros-Nes-v0` / `Level1-1` with profile `supermario-level1-1`.
- Run both benchmark entrypoints:
  - isolated env SPS: `scripts/benchmark_vec_env.py`
  - full training-loop SPS: `scripts/benchmark_sb3_ppo.py`
- Use sampled actions for the primary isolated-env comparison.
- Run a short smoke for each version and each entrypoint before timing.
- Run exactly 3 full timing samples per version and per entrypoint unless the user asks for more. Use the same `--seconds`, `--warmup-steps`, env count, action mode, backend policy, PPO config, device, and package source across comparable runs.
- Prefer 30-second isolated-env timing samples for final env SPS numbers.
- Prefer `--warmup-updates 1 --measured-updates 10 --n-steps 512 --batch-size 512 --n-epochs 4` for final SB3 PPO train-loop numbers unless the user asks for a shorter check.
- Compute arithmetic mean and sample standard deviation for each version and metric.
- Compute speedup as `requested_version_mean_steps_per_second / post0_mean_steps_per_second`, separately for env SPS and train SPS.
- If fixed-action runs are useful for diagnosing noise, report them separately and do not use them for the main speedup unless the user explicitly asks.

The final answer must include two tables, one for isolated env SPS and one for SB3 PPO train SPS, each with at least:

```text
Version/build | Backend | Samples | Mean steps/s | Std steps/s | Speedup vs post0
```

Include the individual sample values below the table or in a compact `samples=[...]` field so variance is auditable.

Keep candidate decisions tied to train SPS. If an optimization improves isolated env SPS but not SB3 PPO train SPS across the 3-sample mean, report it as a diagnostic improvement rather than a keeper for final training throughput.

## Reliability And Load Gate

Before timing, check that the machine is reasonably idle and record the check in the final answer. Do not collect final numbers while the machine is under obvious competing CPU/GPU load.

Recommended local checks:

```bash
python - <<'PY'
import os
print("cpu_count", os.cpu_count())
print("loadavg_1m_5m_15m", os.getloadavg())
PY
ps -Ao pid,pcpu,pmem,comm | sort -k2 -nr | head -15
```

As a rule of thumb, defer final benchmarks if the 1-minute load average is above roughly 75% of logical CPU count before the benchmark starts, if another training/build job is active, or if top CPU processes show unrelated sustained load. A quick smoke can still run under load, but label it as a smoke and do not use it for retained optimization decisions.

## Post0 Baseline Cache

Cache vanilla `stable-retro-turbo==1.0.0.post0` baseline samples after a successful clean run. The cache avoids rerunning an unchanged baseline every time. If the user wants to refresh the baseline, they can delete the matching cache file and rerun the benchmark.

Cache location:

```text
artifacts/benchmark-cache/post0/
```

Cache key must include enough information to prevent mixing incompatible runs:

- package/version: `stable-retro-turbo==1.0.0.post0`
- benchmark entrypoint: `env` or `sb3_ppo`
- profile and resolved preprocessing settings
- env count, thread count, action mode, backend, warmup, and duration for env SPS
- PPO config, warmup updates, measured updates, device, and torch thread count for train SPS
- platform, Python version, and SB3/Torch versions for train SPS

When a matching post0 cache exists, reuse it and clearly label the baseline as cached in the final answer. When no matching cache exists, run the post0 baseline once with the normal 3-sample protocol, write the cache JSON, and then use those samples for the comparison.

After every completed benchmark run, update the latest benchmark comparison
table in `README.md` with the requested/current build first and the `.post0`
baseline second. Keep the table synchronized with the final answer's samples,
mean, sample standard deviation, and speedup multiplier. If the run includes
both env and train-loop benchmarks, the README update must preserve both
metrics or explicitly state why only one table exists.

## Source Of Truth

1. Inspect the current repo state first:

```bash
git status --short --branch
rg -n "supermario-level1-1|megaman-level1" scripts/benchmark_vec_env.json scripts/benchmark_vec_env.py
```

2. Use `scripts/benchmark_vec_env.py` for isolated vector env SPS, `scripts/benchmark_sb3_ppo.py` for full SB3 PPO train-loop SPS, and `scripts/benchmark_vec_env.json` as the profile source.
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
4. For the SB3 PPO training-loop benchmark, use `scripts/benchmark_sb3_ppo.py --package-source checkout` so imports resolve to the current source checkout.

## Wheel Benchmark Workflow

Use this workflow when the user asks for a wheel/release benchmark or when validating artifacts after `$build`.

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

6. For the SB3 PPO training-loop benchmark in a wheel or post0 venv, run `scripts/benchmark_sb3_ppo.py --package-source installed` so the script does not import the source checkout.

## Isolated Env Timing

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

## SB3 PPO Train-Loop Timing

Run a short smoke first:

```bash
"$BENCH_ROOT/venv/bin/python" /absolute/path/to/repo/scripts/benchmark_sb3_ppo.py \
  --package-source installed \
  --num-envs 2 \
  --num-threads 1 \
  --warmup-updates 0 \
  --measured-updates 1 \
  --n-steps 8 \
  --batch-size 16 \
  --n-epochs 1 \
  --device cpu
```

For current-checkout runs, use `.venv314/bin/python` and `--package-source checkout`. For wheel or post0 runs, use the throwaway venv Python and `--package-source installed`.

Then run exactly three full training-loop samples with the same configuration:

```bash
"$BENCH_ROOT/venv/bin/python" /absolute/path/to/repo/scripts/benchmark_sb3_ppo.py \
  --package-source installed \
  --warmup-updates 1 \
  --measured-updates 10 \
  --n-steps 512 \
  --batch-size 512 \
  --n-epochs 4
```

The key metric is `train_steps_per_second` from the JSON result. Also record `rollout_steps_per_second`, `rollout_seconds`, and `update_seconds` so regressions can be attributed to environment stepping versus PPO update overhead.

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
- Machine-load gate result before final timing.
- Individual throughput samples for both isolated env SPS and SB3 PPO train SPS, not only averages.
- For SB3 PPO train-loop runs, include PPO config, device, torch thread count, train SPS, rollout SPS, rollout seconds, and update seconds.
- Whether `.post0` baselines were freshly measured or reused from `artifacts/benchmark-cache/post0/`.
- Confirmation that `README.md` was updated with the latest comparison table.
- A concise comparison to prior remembered baseline only if memory was used and clearly labeled as memory-derived.

Useful interpretation guardrails:

- Mario sampled-action benchmark timing can vary materially by trajectory and system load.
- A clean wheel should not include ROM payloads; import the ROM into the benchmark environment.
- `State.NONE` numbers are emulator hot-path diagnostics, not training-representative results.
- `.post0` is expected to benchmark through `subproc_vec_retro`, not `native_vec_fused`, because it predates `StableRetroNativeVecEnv`.
