---
name: modal-benchmark
description: Run and report the clean-machine Modal CPU benchmark for stable-retro-turbo from either the current checkout (default) or a specific stable-retro-turbo package version. Use when the user invokes /modal-benchmark or $modal-benchmark, asks to benchmark on Modal/modal.com, wants stable benchmark numbers that are not affected by current local Mac CPU load, asks for a clean CPU-only throughput baseline, wants to benchmark a named package version remotely, or needs to validate an optimization on fresh remote compute.
---

# Modal Benchmark

## Overview

Use this skill when local benchmark timing is contaminated by host load, macOS security/indexing churn, or other competing processes. It runs either the current checkout or a named `stable-retro-turbo` package version on provisioned Modal CPU using `scripts/modal_benchmark.py`, then reports structured isolated-env and SB3 PPO train-loop throughput from the saved JSON artifact.

Default to `--package-source checkout`. Use `--package-source version --package-version X.Y.Z.postN` only when the user names a package version or asks to benchmark a published release. Use `benchmark-build` for local wheel-vs-`.post0` comparison work; use this skill when the priority is a repeatable clean-machine target benchmark.

## Workflow

Run from the repository root. First confirm the current branch is the intended branch when using the checkout target:

```bash
git status --short --branch
```

Use the current date in the artifact name. For the default current-checkout target:

```bash
modal run scripts/modal_benchmark.py --output-json artifacts/benchmarks/modal-current-YYYY-MM-DD.json
```

If that path already exists, add a short time suffix:

```bash
modal run scripts/modal_benchmark.py --output-json artifacts/benchmarks/modal-current-YYYY-MM-DD-HHMM.json
```

For a named package version:

```bash
modal run scripts/modal_benchmark.py \
  --package-source version \
  --package-version 1.0.0.postN \
  --output-json artifacts/benchmarks/modal-1.0.0.postN-YYYY-MM-DD.json
```

For a quick remote smoke, use:

```bash
modal run scripts/modal_benchmark.py --smoke --output-json artifacts/benchmarks/modal-smoke-YYYY-MM-DD-HHMM.json
```

Treat an explicit invocation of this skill or a user request for Modal benchmarking as approval to request escalated execution for Modal network/auth/upload. If escalation is blocked, report that the benchmark could not run and plainly name that Modal needs network/auth and uploads a repo snapshot plus the local Mario ROM bytes.

## Defaults

Use the launcher defaults unless the user asks otherwise:

- Modal CPU request: `16.0`
- Modal memory: `16384` MB
- Python: `3.14`
- Package source: `checkout`
- Profile: `supermario-level1-1`
- Game/state: `SuperMarioBros-Nes-v0` / `Level1-1`
- Isolated env samples: `repeats=3`, `env_seconds=30`, `env_warmup_steps=32`
- PPO train samples: `repeats=3`, `warmup_updates=1`, `measured_updates=10`, `n_steps=512`, `batch_size=512`, `n_epochs=4`, `device=cpu`

For `--package-source checkout`, the launcher uploads the current repo snapshot to Modal, builds the extension there, installs the checkout, writes local `stable_retro/data/stable/SuperMarioBros-Nes-v0/rom.nes` bytes into the active package data directory at runtime, and runs:

- `scripts/benchmark_vec_env.py` for isolated env SPS
- `scripts/benchmark_sb3_ppo.py --package-source checkout` for full SB3 PPO train-loop SPS

For `--package-source version`, the launcher still uploads this repo snapshot for the benchmark driver scripts and profile JSON, but the remote function uninstalls the editable checkout package, installs `stable-retro-turbo==PACKAGE_VERSION`, writes the same ROM bytes into that installed package's data directory, and runs:

- `scripts/benchmark_vec_env.py` for isolated env SPS
- `scripts/benchmark_sb3_ppo.py --package-source installed` for full SB3 PPO train-loop SPS

## Reporting

After the run completes, read the saved artifact and report the result. Start with whether it worked and name the target: current checkout with branch/commit, or `stable-retro-turbo==VERSION`. Mention that Modal built the image, uploaded the current checkout for scripts/profile, injected the ROM bytes at runtime, and ran both env and PPO timing samples. For checkout targets, also mention that the native extension was built remotely from the checkout.

Include a file link to the saved artifact.

Include target metadata:

```text
package_source: checkout|version
package_version: VERSION_OR_NONE
branch: BRANCH
commit: COMMIT
```

Include an `Env SPS` table:

```text
Backend | Samples | Mean steps/s | Std steps/s | Best steps/s
native | RUN1, RUN2, RUN3 | MEAN | STDEV | BEST
```

Use the backend/name from `data["env"]["runs"][0]["name"]`.

Include a `SB3 PPO Train SPS` table:

```text
Backend | Samples | Mean steps/s | Std steps/s | Best steps/s
native | RUN1, RUN2, RUN3 | MEAN | STDEV | BEST
```

Use `data["sb3_ppo"]["runs"][i]["timing"]["train_steps_per_second"]`.

Include an attribution block for train-loop components:

```text
rollout_steps_per_second mean: MEAN
rollout_seconds mean: MEAN
update_seconds mean: MEAN
```

Include Modal metadata:

```text
cpu_request: CPU_REQUEST
memory_mb: MEMORY_MB
os_cpu_count: OS_CPU_COUNT
affinity_cpu_count: AFFINITY_CPU_COUNT
stable_retro_file: PATH
stable_retro_extension: PATH
version: VERSION
rom_path: PATH
```

If Modal prints a run URL, include it. If no run URL appears in the command output, omit it rather than inventing one.

Before the final answer, run `git status --short` so changed launcher, skill, or docs files are visible.

## JSON Extraction

Use a local read after the run to avoid hand-copying console output:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/benchmarks/modal-current-YYYY-MM-DD.json")
data = json.loads(path.read_text())

env = data["env"]["summary"]
env_runs = [round(run["steps_per_second"], 1) for run in data["env"]["runs"]]
train = data["sb3_ppo"]["summary"]["train_steps_per_second"]
train_runs = [
    round(run["timing"]["train_steps_per_second"], 1)
    for run in data["sb3_ppo"]["runs"]
]
rollout = data["sb3_ppo"]["summary"]["rollout_steps_per_second"]
rollout_s = data["sb3_ppo"]["summary"]["rollout_seconds"]
update_s = data["sb3_ppo"]["summary"]["update_seconds"]

print("artifact", path)
print("target", data["target"])
print("branch", data["local"]["git"]["branch"])
print("commit", data["local"]["git"]["commit"])
print("env_backend", data["env"]["runs"][0]["name"])
print("env_runs", env_runs)
print("env_mean", env["mean"])
print("env_stdev", env["stdev"])
print("env_best", env["max"])
print("train_backend", data["sb3_ppo"]["runs"][0]["env"]["backend"])
print("train_runs", train_runs)
print("train_mean", train["mean"])
print("train_stdev", train["stdev"])
print("train_best", train["max"])
print("rollout_steps_per_second_mean", rollout["mean"])
print("rollout_seconds_mean", rollout_s["mean"])
print("update_seconds_mean", update_s["mean"])
print("modal", data["modal"])
print("runtime", data["runtime"])
PY
```
