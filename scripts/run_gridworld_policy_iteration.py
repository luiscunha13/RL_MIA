from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.experiments.dynamic_programming import policy_iteration
from mia_rl.mdps.gridworld import GridworldMDP
from mia_rl.plots.gridworld_dp import plot_values_and_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Policy Iteration on tabular Gridworld.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--theta", type=float, default=1e-8, help="Convergence threshold.")
    parser.add_argument("--max-outer", type=int, default=100, help="Maximum number of policy improvement steps.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gridworld_policy_iteration",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def _print_values(env: GridworldMDP, values: dict[tuple[int, int], float]) -> None:
    for r in range(env.n_rows):
        row = [f"{values[(r, c)]:7.2f}" for c in range(env.n_cols)]
        print(" ".join(row))


def _print_policy(env: GridworldMDP, policy: dict[tuple[int, int], str]) -> None:
    for r in range(env.n_rows):
        row: list[str] = []
        for c in range(env.n_cols):
            state = (r, c)
            if env.is_terminal(state):
                row.append(" T ")
            else:
                row.append(f" {policy.get(state, '.')} ")
        print("".join(row))


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    env = GridworldMDP()
    values, policy, history = policy_iteration(env, gamma=args.gamma, theta=args.theta, max_outer=args.max_outer)

    print(f"Policy iteration outer loops: {len(history)}")
    print("\nOptimal values:")
    _print_values(env, values)
    print("\nGreedy policy:")
    _print_policy(env, policy)

    fig, _ = plot_values_and_policy(env, values, policy, title="Gridworld Policy Iteration: V* and greedy policy")

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "gridworld_policy_iteration.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_dir / 'gridworld_policy_iteration.png'}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
