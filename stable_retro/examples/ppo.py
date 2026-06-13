"""Train an agent using PPO with the native stable-retro vector env."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

import stable_retro as retro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis-v0")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()

    venv = VecTransposeImage(
        retro.StableRetroNativeVecEnv(
            args.game,
            num_envs=args.num_envs,
            state=args.state,
            scenario=args.scenario,
            num_threads=args.num_threads,
            render_mode="rgb_array",
            obs_resize=(84, 84),
            obs_grayscale=True,
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=True,
            sticky_action_prob=0.25,
            reward_clip=True,
        ),
    )
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
