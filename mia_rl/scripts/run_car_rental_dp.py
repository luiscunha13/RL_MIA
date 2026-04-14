from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.experiments.dynamic_programming import policy_iteration, value_iteration
from mia_rl.mdps.car_rental import CarRentalMDP, CarRentalParams
from mia_rl.plots.car_rental import plot_policy, plot_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic programming on Jack's Car Rental.")
    parser.add_argument("--max-cars", type=int, default=10, help="Capacity per location.")
    parser.add_argument("--max-move", type=int, default=5, help="Max cars moved overnight.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--theta", type=float, default=1e-4, help="Convergence threshold.")
    parser.add_argument("--mode", choices=("pi", "vi", "both"), default="both", help="Algorithm to run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/car_rental_dp",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")

    plt = importlib.import_module("matplotlib.pyplot")

    params = CarRentalParams(
        max_cars_1=args.max_cars,
        max_cars_2=args.max_cars,
        max_moveable=args.max_move,
    )
    mdp = CarRentalMDP(params)

    if args.mode in ("pi", "both"):
        values_pi, policy_pi, history = policy_iteration(
            mdp,
            gamma=args.gamma,
            theta=args.theta,
            max_outer=20,
            eval_max_iters=200,
        )
        print(f"Policy Iteration outer loops: {len(history)}")
        print(f"Policy Iteration sample V(10,10): {values_pi.get((min(10, args.max_cars), min(10, args.max_cars)), 0.0):.3f}")
        print(f"Policy Iteration policy size: {len(policy_pi)}")

        fig_pi_policy, _ = plot_policy(mdp, policy_pi, title="Car Rental Policy Iteration: policy")
        fig_pi_values, _ = plot_values(mdp, values_pi, title="Car Rental Policy Iteration: V")

        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_pi_policy.savefig(output_dir / "car_rental_pi_policy.png", dpi=150, bbox_inches="tight")
        fig_pi_values.savefig(output_dir / "car_rental_pi_values.png", dpi=150, bbox_inches="tight")

    if args.mode in ("vi", "both"):
        values_vi, policy_vi, iters = value_iteration(
            mdp,
            gamma=args.gamma,
            theta=args.theta,
            max_iters=2_000,
        )
        print(f"Value Iteration iterations: {iters}")
        print(f"Value Iteration sample V(10,10): {values_vi.get((min(10, args.max_cars), min(10, args.max_cars)), 0.0):.3f}")
        print(f"Value Iteration policy size: {len(policy_vi)}")

        fig_vi_policy, _ = plot_policy(mdp, policy_vi, title="Car Rental Value Iteration: greedy policy")
        fig_vi_values, _ = plot_values(mdp, values_vi, title="Car Rental Value Iteration: V*")

        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_vi_policy.savefig(output_dir / "car_rental_vi_policy.png", dpi=150, bbox_inches="tight")
        fig_vi_values.savefig(output_dir / "car_rental_vi_values.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {PACKAGE_ROOT / args.output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
