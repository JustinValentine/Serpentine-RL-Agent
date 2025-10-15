"""Roll out a trained policy in the horizontal cooperative pendulum environment."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from serpentine_rl.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved PPO policy (zip file).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run evaluation in deterministic mode.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between rendered frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(render_mode="human")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env, filename=None)

    model = PPO.load(args.model_path)

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if args.sleep:
                time.sleep(args.sleep)
        print(f"Episode {episode + 1}: reward={total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
