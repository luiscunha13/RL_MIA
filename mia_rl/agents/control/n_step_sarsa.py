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

        oldest_transition = self._pending_transitions[0]

        n_step_window = self._pending_transitions[: self.n_steps]
        rewards_sum = sum(transition.reward * (self.gamma ** i) for i, transition in enumerate(n_step_window))

        if len(n_step_window) == self.n_steps and not n_step_window[-1].done:
            last_transition = n_step_window[-1]
            next_action = self._selected_actions[last_transition.next_state]
            rewards_sum += (self.gamma ** self.n_steps) * self.Q[(last_transition.next_state, next_action)]

        self.Q[(oldest_transition.state, oldest_transition.action)] += self.alpha * (rewards_sum - self.Q[(oldest_transition.state, oldest_transition.action)])

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))