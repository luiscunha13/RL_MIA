from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._pending_transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            action = self.greedy_action(state)

        self._selected_actions[state] = action

        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        self._pending_transitions.append(transition)

        if transition.done:
            while self._pending_transitions:
                self._update_oldest_transition()
                self._pending_transitions.pop(0)
        elif len(self._pending_transitions) >= self.n_steps:
            self._update_oldest_transition()
            self._pending_transitions.pop(0)

    def _update_oldest_transition(self) -> None:
        if not self._pending_transitions:
            return

        oldest_transition = self._pending_transitions[0]

        limit = min(self.n_steps, len(self._pending_transitions))
        rewards_sum = 0.0
        for i in range(limit):
            rewards_sum += (self.gamma ** i) * self._pending_transitions[i].reward

        if limit == self.n_steps:
            last_transition = self._pending_transitions[-1]
            if not last_transition.done and last_transition.next_state is not None:
                next_action = self._selected_actions[last_transition.next_state]
                rewards_sum += (self.gamma ** self.n_steps) * self.Q[(last_transition.next_state, next_action)]

        state_action = (oldest_transition.state, oldest_transition.action)
        self.Q[state_action] += self.alpha * (rewards_sum - self.Q[state_action])

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))