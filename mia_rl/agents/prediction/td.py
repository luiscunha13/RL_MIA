from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        for transition in episode.transitions:
            bootstrap = 0.0 if transition.done or transition.next_state is None else self.V[transition.next_state]
            target = transition.reward + self.gamma * bootstrap
            self.V[transition.state] += self.alpha * (target - self.V[transition.state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])


class NStepTDPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, n: int, alpha: float = 0.05, gamma: float = 1.0):
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        transitions = episode.transitions
        T = len(transitions)
        for t in range(T):
            G = 0.0
            limit = min(t + self.n, T)
            for i in range(t, limit):
                G += (self.gamma ** (i - t)) * transitions[i].reward
            
            if t + self.n < T:
                G += (self.gamma ** self.n) * self.V[transitions[t + self.n].state]
            else:
                last_trans = transitions[-1]
                if not last_trans.done and last_trans.next_state is not None:
                    G += (self.gamma ** (T - t)) * self.V[last_trans.next_state]
            
            self.V[transitions[t].state] += self.alpha * (G - self.V[transitions[t].state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
