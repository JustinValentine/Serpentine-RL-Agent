"""Generate animated GIF rollouts from a trained PPO policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

os.environ.setdefault("MUJOCO_GL", "egl")

from serpentine_rl.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save GIF rollouts for the horizontal pendulum policy."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved PPO policy ZIP file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="renders",
        help="Directory where GIF files will be written.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollouts to record.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Render with a deterministic policy evaluation.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Playback frame rate for the saved GIFs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for environment resets.",
    )
    return parser.parse_args()


def _collect_frames(
    env: gym.Env,
    model: PPO,
    deterministic: bool,
    max_steps: Optional[int],
    seed: int,
) -> List:
    frames: List = []
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    while not done and (max_steps is None or step < max_steps):
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "Environment did not return an RGB frame. "
                "Ensure render_mode='rgb_array' is supported."
            )
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    return frames


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)
    env = Monitor(env, filename=None)

    model = PPO.load(args.model_path)

    for episode in range(args.episodes):
        seed = args.seed + episode
        frames = _collect_frames(
            env=env,
            model=model,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            seed=seed,
        )

        gif_path = output_dir / f"episode_{episode + 1:02d}.gif"
        imageio.mimsave(gif_path, frames, duration=1.0 / args.fps)
        print(f"Wrote {gif_path} ({len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
