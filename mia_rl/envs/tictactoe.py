from __future__ import annotations

from mia_rl.core.base import Environment

# ── Type aliases ────────────────────────────────────────────────────────────
# The board is a 9-tuple of ints (one per cell, row-major):
#   0 = empty, 1 = player X, -1 = player O
# Actions are integers 0-8 identifying the cell to mark.
TicTacToeState  = tuple[int, ...]   # length-9
TicTacToeAction = int               # 0 … 8

# Indices of every winning line (rows, columns, diagonals)
_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def _winner(board: TicTacToeState) -> int:
    """Return 1 if X wins, -1 if O wins, 0 otherwise."""
    for i, j, k in _WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class TicTacToeEnv(Environment[TicTacToeState, TicTacToeAction]):
    """Two-player Tic-Tac-Toe environment.

    Conventions:
    - Player X always goes first (represented as +1 in the board).
    - Player O is represented as -1.
    - `current_player` alternates between 1 (X) and -1 (O) each step.
    - The state is a length-9 tuple representing all 9 cells row-major:
        indices  0 1 2
                 3 4 5
                 6 7 8
    - `step()` applies the current player's move, then switches turns.
    - Episode ends when a player wins or the board is full (draw).
    - Rewards from the perspective of the player who just moved:
        +1  for winning
        -1  for losing (opponent wins — not possible in one step, included for completeness)
         0  otherwise (ongoing or draw)

    For self-play, call `reset()` at the start of each game and alternate
    calling `step()` for player X and player O.
    """

    def __init__(self) -> None:
        self.board: TicTacToeState = (0,) * 9
        self.current_player: int = 1  # X starts

    def reset(self) -> TicTacToeState:
        """Reset the board to an empty state and set X as the first player."""

        self.board = (0,) * 9
        self.current_player = 1     
        return self.board

    def available_actions(self, state: TicTacToeState) -> list[TicTacToeAction]:
        """Return the indices of all empty cells in `state`."""

        return [i for i, cell in enumerate(state) if cell == 0]

    def is_terminal(self, state: TicTacToeState) -> bool:
        """Return True if the game is over (win or draw)."""

        return _winner(state) != 0 or all(cell != 0 for cell in state)

    def step(self, action: TicTacToeAction) -> tuple[TicTacToeState, float, bool]:
        """Place the current player's mark on cell `action` and advance the game."""

        if self.board[action] != 0:
            raise ValueError(f"Illegal action {action}: cell is not empty")

        new_board = list(self.board)
        new_board[action] = self.current_player
        new_board = tuple(new_board)

        winner = _winner(new_board)
        done = winner != 0 or all(cell != 0 for cell in new_board)
        reward = 1.0 if winner == self.current_player else 0.0

        self.current_player = -self.current_player 
        self.board = new_board
        return new_board, reward, done

    def render(self, state: TicTacToeState | None = None) -> None:
        """Print a human-readable board to stdout.

        Uses a fixed mapping so symbols are consistent across turns:
        - X for board value 1
        - O for board value -1
        - cell index (0-8) for empty cells
        """
        board = state if state is not None else self.board
        rows = []

        def cell_symbol(idx: int, value: int) -> str:
            if value == 1:
                return "X"
            if value == -1:
                return "O"
            return str(idx)

        for r in range(3):
            row_cells = []
            for c in range(3):
                idx = r * 3 + c
                row_cells.append(f" {cell_symbol(idx, board[idx])} ")
            rows.append("|".join(row_cells))

        divider = "---+---+---"
        print(rows[0])
        print(divider)
        print(rows[1])
        print(divider)
        print(rows[2])
