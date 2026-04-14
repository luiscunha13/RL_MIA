from __future__ import annotations

import importlib
import numpy as np

from mia_rl.mdps.car_rental import CarRentalMDP


def _plt_module():
    return importlib.import_module("matplotlib.pyplot")


def policy_to_array(mdp: CarRentalMDP, policy: dict[tuple[int, int], int]) -> np.ndarray:
    arr = np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=int)
    for (n1, n2), action in policy.items():
        arr[n1, n2] = action
    return arr


def values_to_array(mdp: CarRentalMDP, values: dict[tuple[int, int], float]) -> np.ndarray:
    arr = np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=float)
    for (n1, n2), value in values.items():
        arr[n1, n2] = value
    return arr


def plot_policy(mdp: CarRentalMDP, policy: dict[tuple[int, int], int], title: str = "Policy"):
    plt = _plt_module()
    arr = policy_to_array(mdp, policy)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("# cars at location 2")
    ax.set_ylabel("# cars at location 1")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig, ax


def plot_values(mdp: CarRentalMDP, values: dict[tuple[int, int], float], title: str = "State values"):
    plt = _plt_module()
    arr = values_to_array(mdp, values)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("# cars at location 2")
    ax.set_ylabel("# cars at location 1")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig, ax
