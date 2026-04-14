from __future__ import annotations

from mia_rl.core.base import Environment

WindyGridworldState = tuple[int, int]
WindyGridworldAction = str

ACTIONS: tuple[WindyGridworldAction, ...] = ("up", "down", "left", "right")
ACTION_TO_DELTA: dict[WindyGridworldAction, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class WindyGridworldEnv(Environment[WindyGridworldState, WindyGridworldAction]):
    def __init__(
        self,
        rows: int = 7,
        cols: int = 10,
        start: WindyGridworldState = (3, 0),
        goal: WindyGridworldState = (3, 7),
        wind: tuple[int, ...] = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0),
        reward_per_step: float = -1.0,
    ):
        if len(wind) != cols:
            raise ValueError("The wind specification must contain one strength per column.")

        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.wind = wind
        self.reward_per_step = reward_per_step
        self.current_state = start

    def states(self) -> list[WindyGridworldState]:
        return [(row, col) for row in range(self.rows) for col in range(self.cols)]

    def reset(self) -> WindyGridworldState:
        self.current_state = self.start
        return self.current_state

    def available_actions(self, state: WindyGridworldState) -> list[WindyGridworldAction]:
        return list(ACTIONS)

    def step_from_state(
        self,
        state: WindyGridworldState,
        action: WindyGridworldAction,
    ) -> tuple[WindyGridworldState, float, bool]:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        row, col = state

        delta_row, delta_col = ACTION_TO_DELTA[action]
        delta_row -= self.wind[col]
        next_row = max(0, min(self.rows - 1, row + delta_row))
        next_col = max(0, min(self.cols - 1, col + delta_col))
        next_state = (next_row, next_col)
        return next_state, self.reward_per_step, next_state == self.goal

    def step(self, action: WindyGridworldAction) -> tuple[WindyGridworldState, float, bool]:
        next_state, reward, done = self.step_from_state(self.current_state, action)
        self.current_state = next_state
        return next_state, reward, done
