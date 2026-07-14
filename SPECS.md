## PROJECT PURPOSE

`stable-retro-turbo` gives reinforcement-learning developers a high-throughput way to run classic-console games in batches while retaining Stable Retro's integrations and single-environment behavior. It must preserve faithful, controllable rollouts through the upstream-compatible Python API and a Gymnasium-native vector API, with users supplying any required ROMs.

## PROJECT REQUIREMENTS

- The project must preserve Stable Retro's supported game integrations and public single-environment semantics, including saved-state loading, scenario rewards and termination, raw info variables, and ROM import; Turbo features must not reinterpret or remove them.
- Published package artifacts must exclude ROM payloads and allow users to supply required ROMs through the supported import workflow.
- `RetroVecEnv` must implement Gymnasium's vector-environment contract directly and expose batched observations, rewards, terminations, truncations, and infos without requiring Stable-Baselines3.
- Each `RetroVecEnv` lane must preserve the corresponding `RetroEnv` emulator, action, saved-state, scenario reward and termination, and info semantics before configured vector-only transforms.
- `RetroVecEnv` must use disabled autoreset: terminal lanes retain their terminal observations and cannot be stepped until selected for reset, while a masked reset leaves every unselected lane's emulator state, random stream, and observation history unchanged.
- Per-lane saved-state starts and seeded state selection must be reproducible for identical inputs.
- Non-benchmark observation ownership modes must not expose results that a later environment call can unexpectedly mutate.
