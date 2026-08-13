"""Native vector rollout example.

SB3 adapters are intentionally downstream; use RetroVecEnv as a Gymnasium
VectorEnv and adapt it in the training project that owns SB3 integration.
"""

import argparse

import stable_retro as retro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis-v0")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    env = retro.RetroVecEnv(
        args.game,
        num_envs=args.num_envs,
        state=args.state,
        scenario=args.scenario,
        num_threads=args.num_threads,
        render_mode="rgb_array",
        obs_copy="copy",
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="nearest",
        obs_layout="chw",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        sticky_action_prob=0.25,
        reward_clip=True,
    )
    try:
        obs, infos = env.reset(seed=123)
        for _ in range(args.steps):
            obs, rewards, terminations, truncations, infos = env.step(
                env.action_space.sample(),
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
