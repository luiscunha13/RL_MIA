from __future__ import annotations

import numpy as np


class BanditAgent:
    """Base class for Multi-armed Bandit agents."""

    def __init__(self, k: int = 10):
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError


class EpsilonGreedy(BanditAgent):
    """Epsilon-greedy bandit agent with optimistic initialization support."""

    def __init__(self, k: int = 10, epsilon: float = 0.1, optimistic: float = 0.0):
        self.epsilon = epsilon
        self.optimistic = optimistic
        super().__init__(k=k)

    def reset(self) -> None:
        super().reset()
        self.Q[:] = self.optimistic

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.k))
        else:
            return int(np.argmax(self.Q))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class UCB(BanditAgent):
    """Upper Confidence Bound bandit agent."""

    def __init__(self, k: int = 10, c: float = 2.0):
        self.c = c
        super().__init__(k=k)

    def select_action(self) -> int:
        self.t += 1
        # Try each action once first
        for a in range(self.k):
            if self.N[a] == 0:
                return a
        # UCB selection
        return int(np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N)))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class GradientBandit(BanditAgent):
    """Gradient Bandit agent (preference-based)."""

    def __init__(self, k: int = 10, alpha: float = 0.1, baseline: bool = True):
        self.alpha = alpha
        self.baseline = baseline
        super().__init__(k=k)

    def reset(self) -> None:
        super().reset()
        self.H = np.zeros(self.k)
        self.avg_reward = 0.0

    def _policy(self) -> np.ndarray:
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self) -> int:
        probs = self._policy()
        return int(np.searchsorted(np.cumsum(probs), np.random.random()))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        probs = self._policy()

        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0.0

        diff = reward - baseline
        self.H -= self.alpha * diff * probs
        self.H[action] += self.alpha * diff
