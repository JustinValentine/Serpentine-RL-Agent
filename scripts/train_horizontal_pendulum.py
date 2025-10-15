"""Training entry-point for the horizontal cooperative pendulum."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from serpentine_rl.envs import make_env


def _build_env(seed: int, render: bool = False) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = make_env(render_mode="human" if render else None)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cooperative agent on the horizontal pendulum."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=150_000,
        help="Number of environment interaction steps for training.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S")),
        help="Directory to store tensorboard logs and checkpoints.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10_000,
        help="How often (in steps) to run evaluation episodes.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per evaluation round.",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Policy optimizer learning rate.",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Enable human rendering during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)

    train_env = DummyVecEnv([_build_env(seed=args.seed)])

    eval_env = DummyVecEnv(
        [_build_env(seed=args.seed + 1, render=args.render_eval)]
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.log_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_frequency,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=args.render_eval,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=512,
        n_steps=4_096,
        ent_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        n_epochs=12,
        clip_range_vf=None,
        target_kl=0.06,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(os.path.join(args.log_dir, "horizontal_pendulum_ppo"))


if __name__ == "__main__":
    main()
