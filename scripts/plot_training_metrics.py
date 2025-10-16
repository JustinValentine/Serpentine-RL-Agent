#!/usr/bin/env python3
"""
Utility to plot episode length and return curves from TensorBoard logs.

Example:
    python scripts/plot_training_metrics.py \
        --log-dir runs/horizontal_pendulum/PPO_26 \
        --output-dir renders/training_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _resolve_event_file(log_dir: Path) -> Path:
    if log_dir.is_file():
        return log_dir
    event_files = sorted(
        log_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {log_dir}")
    return event_files[0]


def _load_series(event_path: Path, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    try:
        scalars = accumulator.Scalars(tag)
    except KeyError as exc:
        raise KeyError(f"TensorBoard tag '{tag}' not found in {event_path}") from exc
    if not scalars:
        raise ValueError(f"No scalar data for '{tag}' in {event_path}")
    steps = np.array([scalar.step for scalar in scalars], dtype=np.float64)
    values = np.array([scalar.value for scalar in scalars], dtype=np.float64)
    return steps, values


def _plot_series(
    steps: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps / 1e6, values, lw=1.5)
    plt.xlabel("Total Timesteps (millions)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot episode statistics from Stable-Baselines3 TensorBoard logs."
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Directory containing TensorBoard event files, or a path to a specific event file.",
    )
    parser.add_argument(
        "--output-dir",
        default="renders/training_plots",
        help="Directory where PNG plots will be saved (created if missing).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    log_path = Path(args.log_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    event_file = _resolve_event_file(log_path)

    for tag, ylabel, title, filename in (
        ("rollout/ep_len_mean", "Episode Length (steps)", "Episode Length vs Timesteps", "ep_len_mean.png"),
        ("rollout/ep_rew_mean", "Episode Return", "Episode Return vs Timesteps", "ep_rew_mean.png"),
    ):
        steps, values = _load_series(event_file, tag)
        output_path = output_dir / filename
        _plot_series(steps, values, ylabel, title, output_path)
        print(f"Saved {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
