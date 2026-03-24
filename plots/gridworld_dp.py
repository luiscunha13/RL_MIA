from __future__ import annotations

import importlib
import numpy as np

from mia_rl.mdps.gridworld import GridworldMDP

ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}


def _plt_module():
    return importlib.import_module("matplotlib.pyplot")


def values_to_array(env: GridworldMDP, values: dict[tuple[int, int], float]) -> np.ndarray:
    arr = np.zeros((env.n_rows, env.n_cols), dtype=float)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            arr[r, c] = values.get((r, c), 0.0)
    return arr


def plot_values_and_policy(
    env: GridworldMDP,
    values: dict[tuple[int, int], float],
    policy: dict[tuple[int, int], str] | None = None,
    title: str = "Gridworld values",
):
    plt = _plt_module()
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_title(title)

    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(env.n_cols + 1))
    ax.set_yticks(np.arange(env.n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for r, c in env.terminal_states:
        rect = plt.Rectangle((c, r), 1, 1, fill=True, alpha=0.15)
        ax.add_patch(rect)

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state = (r, c)
            ax.text(c + 0.5, r + 0.45, f"{values.get(state, 0.0):.2f}", ha="center", va="center", fontsize=11)
            if policy is not None and not env.is_terminal(state):
                action = policy.get(state)
                if action is not None:
                    ax.text(c + 0.5, r + 0.78, ARROW.get(action, "·"), ha="center", va="center", fontsize=16)

    return fig, ax
