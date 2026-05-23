from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.envs.kbandits import KArmedBandit
from mia_rl.agents.kbandits import EpsilonGreedy, UCB, GradientBandit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-Armed Bandit experiments.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps per run.")
    parser.add_argument("--runs", type=int, default=2000, help="Number of independent runs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/kbandits",
        help="Directory inside mia_rl where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def run_experiment(agent, env, steps=1000, runs=2000):
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))

    for r in range(runs):
        env.reset()
        agent.reset()

        for t in range(steps):
            action = agent.select_action()
            _, reward, _ = env.step(action)
            agent.update(action, reward)

            rewards[r, t] = reward
            optimal[r, t] = (action == env.optimal_action)

    return rewards.mean(axis=0), optimal.mean(axis=0)


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    steps = args.steps
    runs = args.runs
    env = KArmedBandit()

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Epsilon Greedy
    print(f"Running epsilon-greedy experiment ({runs} runs, {steps} steps)...")
    epsilons = [0, 0.01, 0.1]
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for eps in epsilons:
        agent = EpsilonGreedy(epsilon=eps)
        rewards, optimal = run_experiment(agent, env, steps, runs)
        ax1.plot(rewards, label=f"e={eps}")
        ax2.plot(optimal * 100, label=f"e={eps}")

    ax1.set_ylabel("Average reward")
    ax1.set_title("e-Greedy: Average Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_title("e-Greedy: % Optimal Action")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "epsilon_greedy.png", dpi=150)

    # 2. Optimistic vs UCB
    print(f"Running optimistic vs UCB experiment ({runs} runs, {steps} steps)...")
    agents = {
        "Optimistic greedy (Q0=5, e=0)": EpsilonGreedy(epsilon=0, optimistic=5),
        "UCB (c=2)": UCB(c=2),
        "Realistic greedy (Q0=0, e=0.1)": EpsilonGreedy(epsilon=0.1, optimistic=0),
    }
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, agent in agents.items():
        rewards, optimal = run_experiment(agent, env, steps, runs)
        ax1.plot(rewards, label=name)
        ax2.plot(optimal * 100, label=name)

    ax1.set_ylabel("Average reward")
    ax1.set_title("Optimistic vs UCB: Average Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_title("Optimistic vs UCB: % Optimal Action")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "optimistic_vs_ucb.png", dpi=150)

    # 3. Gradient Bandit
    print(f"Running gradient bandit experiment ({runs} runs, {steps} steps)...")
    gradient_agents = {
        "alpha=0.1 with baseline": GradientBandit(alpha=0.1, baseline=True),
        "alpha=0.4 with baseline": GradientBandit(alpha=0.4, baseline=True),
        "alpha=0.1 no baseline": GradientBandit(alpha=0.1, baseline=False),
        "alpha=0.4 no baseline": GradientBandit(alpha=0.4, baseline=False),
    }
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, agent in gradient_agents.items():
        rewards, optimal = run_experiment(agent, env, steps, runs)
        ax1.plot(rewards, label=name)
        ax2.plot(optimal * 100, label=name)

    ax1.set_ylabel("Average reward")
    ax1.set_title("Gradient Bandit: Average Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_title("Gradient Bandit: % Optimal Action")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(output_dir / "gradient_bandit.png", dpi=150)

    print(f"All plots saved to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
