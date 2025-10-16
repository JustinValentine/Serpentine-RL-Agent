"""Evaluate a trained policy and save animated GIF rollouts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

os.environ.setdefault("MUJOCO_GL", "egl")

from serpentine_rl.envs import make_env


def _resolve_model_path(path_str: str) -> Path:
    """Resolve model path, accepting directories or missing '.zip' suffixes."""
    path = Path(path_str)

    if path.is_dir():
        zip_candidates = sorted(path.glob("*.zip"))
        if not zip_candidates:
            raise FileNotFoundError(
                f"No .zip checkpoints found in directory '{path}'."
            )
        for preferred in ("best_model.zip", "latest.zip", "horizontal_pendulum_ppo.zip"):
            for candidate in zip_candidates:
                if candidate.name == preferred:
                    return candidate
        return zip_candidates[0]

    candidates = [path]
    if path.suffix != ".zip":
        candidates.append(path.with_suffix(path.suffix + ".zip" if path.suffix else ".zip"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    parent = path.parent if path.parent != Path() else Path(".")
    available = sorted(p.name for p in parent.glob("*.zip"))
    available_msg = ", ".join(available) if available else "none"
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find model checkpoint. Tried: {tried}. "
        f"Available zip files in '{parent}': {available_msg}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a horizontal pendulum policy and save GIFs."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved PPO policy ZIP file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation rollouts.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="renders",
        help="Directory where GIFs will be written.",
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
        help="Maximum steps per episode. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Base seed; each episode adds its index to this value.",
    )
    return parser.parse_args()


def _rollout_episode(
    env: gym.Env,
    model: PPO,
    deterministic: bool,
    max_steps: Optional[int],
    seed: Optional[int],
) -> Tuple[List[np.ndarray], float, int]:
    frames: List[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step_count = 0

    while not done and (max_steps is None or step_count < max_steps):
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "Environment did not produce an RGB frame. "
                "Ensure render_mode='rgb_array' is supported."
            )
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        step_count += 1

    return frames, total_reward, step_count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    max_episode_steps = None if args.max_steps == 0 else args.max_steps
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env, filename=None)

    model_path = _resolve_model_path(args.model_path)
    model = PPO.load(model_path)

    for episode in range(args.episodes):
        seed = args.seed + episode if args.seed is not None else None
        frames, total_reward, steps = _rollout_episode(
            env=env,
            model=model,
            deterministic=args.deterministic,
            max_steps=max_episode_steps,
            seed=seed,
        )

        gif_path = output_dir / f"episode_{episode + 1:02d}.gif"
        imageio.mimsave(gif_path, frames, duration=1.0 / args.fps)
        print(
            f"Episode {episode + 1}: reward={total_reward:.2f} steps={steps} -> {gif_path}"
        )

    env.close()


if __name__ == "__main__":
    main()
