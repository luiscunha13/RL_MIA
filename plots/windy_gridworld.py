from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mia_rl.envs.windy_gridworld import WindyGridworldAction, WindyGridworldEnv, WindyGridworldState

ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}


def plot_episode_lengths(lengths: list[int], title: str = "Episode length over training"):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(lengths)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title(title)
    return fig, ax


def plot_episode_rewards(rewards: list[float], title: str = "Episode reward over training"):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    return fig, ax


def plot_policy(
    env: WindyGridworldEnv,
    policy: dict[WindyGridworldState, WindyGridworldAction],
    path: list[WindyGridworldState] | None = None,
    title: str = "Learned greedy policy",
):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(env.cols + 1))
    ax.set_yticks(np.arange(env.rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            wind_strength = env.wind[col]
            ax.text(col + 0.15, row + 0.2, str(wind_strength), fontsize=8, alpha=0.6)
            if state == env.goal:
                ax.text(col + 0.5, row + 0.55, "G", ha="center", va="center", fontsize=14)
                continue
            if state == env.start:
                ax.text(col + 0.25, row + 0.55, "S", ha="center", va="center", fontsize=14)
            action = policy.get(state)
            if action is not None:
                ax.text(col + 0.6, row + 0.55, ARROWS[action], ha="center", va="center", fontsize=16)

    if path is not None and len(path) > 1:
        xs = [col + 0.5 for _, col in path]
        ys = [row + 0.5 for row, _ in path]
        ax.plot(xs, ys, color="tab:red", linewidth=2, marker="o", markersize=4)

    return fig, ax
