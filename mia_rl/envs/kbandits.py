from __future__ import annotations

import numpy as np

from mia_rl.core.base import Environment


class KArmedBandit(Environment[int, int]):
    """K-Armed Bandit environment.

    Supports both stationary and non-stationary action values.
    """

    def __init__(self, k: int = 10, stationary: bool = True, walk_std: float = 0.01):
        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std
        self.reset()

    def reset(self) -> int:
        self.q_true = np.random.randn(self.k)  # true action values
        self.optimal_action = np.argmax(self.q_true)
        return 0  # Dummy state

    def available_actions(self, state: int) -> list[int]:
        return list(range(self.k))

    def step(self, action: int) -> tuple[int, float, bool]:
        reward = float(np.random.randn() + self.q_true[action])

        # non-stationary random walk
        if not self.stationary:
            self.q_true += np.random.normal(0, self.walk_std, self.k)
            self.optimal_action = np.argmax(self.q_true)

        return 0, reward, False  # Dummy next state, reward, done=False (infinite horizon/single step)

    def is_terminal(self, state: int) -> bool:
        return False
