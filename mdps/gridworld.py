from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mia_rl.mdps.base import TabularMDP

GridState = tuple[int, int]
GridAction = str

ACTIONS: tuple[GridAction, ...] = ("U", "D", "L", "R")

_ACTION_TO_DELTA: dict[GridAction, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(frozen=True)
class GridworldMDP(TabularMDP[GridState, GridAction]):
    n_rows: int = 4
    n_cols: int = 4
    terminal_states: tuple[GridState, ...] = ((0, 0), (3, 3))
    step_reward: float = -1.0

    def states(self) -> list[GridState]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def possible_actions(self, state: GridState) -> list[GridAction]:
        if self.is_terminal(state):
            return []
        return list(ACTIONS)

    def is_terminal(self, state: GridState) -> bool:
        return state in self.terminal_states

    def transitions(
        self,
        state: GridState,
        action: GridAction,
    ) -> Iterable[tuple[float, GridState, float, bool]]:
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        dr, dc = _ACTION_TO_DELTA[action]
        nr, nc = state[0] + dr, state[1] + dc

        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            next_state = state
        else:
            next_state = (nr, nc)

        done = self.is_terminal(next_state)
        return [(1.0, next_state, self.step_reward, done)]
