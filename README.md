# MIA Reinforcement Learning

This repository contains the codebase for the Reinforcement Learning course. It includes implementations of various RL algorithms, from Dynamic Programming to Policy Gradients.

## Package Organization

The core logic resides in the `mia_rl/` package:

- [core/](file:///home/luiscunha/RL_MIA/mia_rl/core/) — Generic abstractions (`Environment`, `Agent`, `Policy`, etc.)
- [envs/](file:///home/luiscunha/RL_MIA/mia_rl/envs/) — Environments (e.g., Windy Gridworld, TicTacToe, Blackjack)
- [mdps/](file:///home/luiscunha/RL_MIA/mia_rl/mdps/) — MDP abstractions for Dynamic Programming
- [agents/](file:///home/luiscunha/RL_MIA/mia_rl/agents/) — RL algorithms (SARSA, REINFORCE, Monte Carlo, etc.)
- [features/](file:///home/luiscunha/RL_MIA/mia_rl/features/) — State representation and feature engineering
- [policies/](file:///home/luiscunha/RL_MIA/mia_rl/policies/) — Reusable policy implementations
- [experiments/](file:///home/luiscunha/RL_MIA/mia_rl/experiments/) — Training and evaluation loops
- [notebooks/](file:///home/luiscunha/RL_MIA/mia_rl/notebooks/) — Interactive tutorials and practicals
- [scripts/](file:///home/luiscunha/RL_MIA/mia_rl/scripts/) — Executable experiment scripts
- [plots/](file:///home/luiscunha/RL_MIA/mia_rl/plots/) — Visualization helpers
- [outputs/](file:///home/luiscunha/RL_MIA/mia_rl/outputs/) — Saved results and plots

## Setup

1. **Create the environment**:
   ```bash
   conda env create -f mia_rl/environment.yml
   ```
2. **Activate the environment**:
   ```bash
   conda activate rl
   ```

## Running Experiments

Experiments can be run as Python modules from the project root:

- **Windy Gridworld (SARSA)**:
  ```bash
  python -m mia_rl.scripts.run_windy_gridworld_sarsa
  ```
- **TicTacToe (Policy Gradient)**:
  See [notebooks/TicTacToe_PolicyGradient.ipynb](file:///home/luiscunha/RL_MIA/mia_rl/notebooks/TicTacToe_PolicyGradient.ipynb).
