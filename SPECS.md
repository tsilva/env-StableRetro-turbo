## PROJECT PURPOSE

`env-StableRetro-turbo` gives reinforcement-learning developers a high-throughput way to run classic-console games in batches while retaining Stable Retro's integrations and single-environment behavior. It must preserve faithful, controllable rollouts through the upstream-compatible Python API and a Gymnasium-native vector API, with users supplying any required ROMs.

## PROJECT REQUIREMENTS

- Publish the Python distribution as `env-stableretro-turbo` while preserving the public `stable_retro` import package.
- The project must preserve Stable Retro's supported game integrations and public single-environment semantics, including saved-state loading, scenario rewards and termination, raw info variables, and ROM import; Turbo features must not reinterpret or remove them.
- Scalar and vector environments must accept every Stable Retro built-in action mode, exact caller-supplied button-label action tables, and game-owned named tables with identical ordered controller semantics.
- Published package artifacts must exclude ROM payloads and allow users to supply required ROMs through the supported import workflow.
- The interactive CLI must launch any imported game by ID and apply supported startup configuration through command-line options without requiring user-written Python.
- `RetroVecEnv` must implement Gymnasium's vector-environment contract directly and expose batched observations, rewards, terminations, truncations, and infos without requiring Stable-Baselines3.
- Each `RetroVecEnv` lane must preserve the corresponding `RetroEnv` emulator, action, saved-state, scenario reward and termination, and info semantics before configured vector-only transforms.
- `RetroVecEnv` must use disabled autoreset: terminal lanes retain their terminal observations and cannot be stepped until selected for reset, while a masked reset leaves every unselected lane's emulator state, random stream, and observation history unchanged.
- Saved-state catalogs must preserve declared order, and each selected lane must reset to an explicit caller-selected catalog index so execution is reproducible without provider-owned sampling.
- Snapshot-capable `RetroVecEnv` instances must capture live lanes and restore selected lanes from reusable same-instance snapshots without advancing emulation or changing unselected lanes; capability must report unavailable when exact core and scenario state cannot be preserved.
- Non-benchmark observation ownership modes must not expose results that a later environment call can unexpectedly mutate.
- Require releases to pass ROM-backed, bit-exact differential validation against the pinned original Stable Retro release for canonical Super Mario Bros and Breakout scalar and vector workloads, covering native and processed observations, actions, rewards, episode boundaries, emulator RAM, shared info signals, lane behavior, and snapshot-restored continuation.
- Publish binary distributions only for Apple-silicon macOS and x86-64 Linux.
