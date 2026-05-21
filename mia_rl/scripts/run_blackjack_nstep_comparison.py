from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Blackjack model-free prediction algorithms.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size alpha.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2, 4, 8], help="List of n values for n-step TD.")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack_prediction_comparison", help="Directory inside mia_rl where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def calculate_mse(val_a: dict, val_b: dict) -> float:
    squared_errors = []
    for state in val_a:
        if state in val_b:
            squared_errors.append((val_a[state] - val_b[state]) ** 2)
    return float(np.mean(squared_errors)) if squared_errors else 0.0


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from mia_rl.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction, NStepTDPrediction
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.experiments.training import train_prediction_agent
    from mia_rl.plots.blackjack import plot_value_difference, plot_value_function
    from mia_rl.policies.blackjack import ThresholdPolicy

    policy = ThresholdPolicy(threshold=args.threshold)
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # MC Agent (baseline)
    print(f"Training First-Visit Monte Carlo for {args.episodes} episodes...")
    mc_env = BlackjackEnv(seed=args.seed)
    mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
    mc_history = train_prediction_agent(mc_env, policy, mc_agent, args.episodes)
    final_mc = mc_history[args.episodes]

    fig_mc, _ = plot_value_function(final_mc, title=f"First-Visit MC after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
    fig_mc.savefig(output_dir / "blackjack_mc.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mc)

    # TD(0) Agent
    print(f"Training TD(0) for {args.episodes} episodes...")
    td_env = BlackjackEnv(seed=args.seed)
    td_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)
    td_history = train_prediction_agent(td_env, policy, td_agent, args.episodes)
    final_td = td_history[args.episodes]

    td_mse = calculate_mse(final_td, final_mc)
    print(f"TD(0) vs MC MSE: {td_mse:.6f}")

    fig_td, _ = plot_value_function(final_td, title=f"TD(0) after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
    fig_td.savefig(output_dir / "blackjack_td0.png", dpi=150, bbox_inches="tight")
    plt.close(fig_td)

    fig_diff_td, _ = plot_value_difference(final_td, final_mc, title=f"TD(0) - MC Difference (MSE: {td_mse:.5f})", vmin=-0.5, vmax=0.5)
    fig_diff_td.savefig(output_dir / "diff_td0_mc.png", dpi=150, bbox_inches="tight")
    plt.close(fig_diff_td)

    # NStepTD Agents
    for n in args.n_values:
        print(f"Training {n}-step TD prediction for {args.episodes} episodes...")
        nstep_env = BlackjackEnv(seed=args.seed)
        nstep_agent = NStepTDPrediction(n=n, alpha=args.td_alpha, gamma=1.0)
        nstep_history = train_prediction_agent(nstep_env, policy, nstep_agent, args.episodes)
        final_nstep = nstep_history[args.episodes]

        nstep_mse = calculate_mse(final_nstep, final_mc)
        print(f"{n}-step TD vs MC MSE: {nstep_mse:.6f}")

        fig_nstep, _ = plot_value_function(final_nstep, title=f"{n}-step TD after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
        fig_nstep.savefig(output_dir / f"blackjack_nstep_n{n}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_nstep)

        fig_diff_nstep, _ = plot_value_difference(final_nstep, final_mc, title=f"{n}-step TD - MC (MSE: {nstep_mse:.5f})", vmin=-0.5, vmax=0.5)
        fig_diff_nstep.savefig(output_dir / f"diff_n{n}_mc.png", dpi=150, bbox_inches="tight")
        plt.close(fig_diff_nstep)

    print(f"\nAll plots saved to {output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
