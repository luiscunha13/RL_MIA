# MIA Reinforcement Learning

This repository contains the codebase for the Reinforcement Learning course. It includes implementations of various RL algorithms, from Dynamic Programming to Policy Gradients.

## Package Organization

The core logic resides in the `mia_rl/` package:

- [core/](mia_rl/core/) — Generic abstractions (`Environment`, `Agent`, `Policy`, etc.)
- [envs/](mia_rl/envs/) — Environments (e.g., Windy Gridworld, TicTacToe, Blackjack)
- [mdps/](mia_rl/mdps/) — MDP abstractions for Dynamic Programming
- [agents/](mia_rl/agents/) — RL algorithms (SARSA, REINFORCE, Monte Carlo, prediction and planning)
- [features/](mia_rl/features/) — State representation and feature engineering
- [policies/](mia_rl/policies/) — Reusable policy implementations
- [experiments/](mia_rl/experiments/) — Training and evaluation loops
- [notebooks/](mia_rl/notebooks/) — Interactive tutorials and practicals
- [scripts/](mia_rl/scripts/) — Executable experiment scripts
- [plots/](mia_rl/plots/) — Visualization helpers
- [outputs/](mia_rl/outputs/) — Saved results and plots

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

- **Windy Gridworld**:
  ```bash
  python -m mia_rl.scripts.run_windy_gridworld_sarsa
  ```
- **Other available scripts**:
   - `python -m mia_rl.scripts.run_windy_gridworld_n_step_sarsa`
   - `python -m mia_rl.scripts.run_windy_gridworld_mc_control`
   - `python -m mia_rl.scripts.run_windy_gridworld_linear_sarsa`
   - `python -m mia_rl.scripts.run_windy_gridworld_linear_td`
   - `python -m mia_rl.scripts.run_windy_gridworld_torch_sarsa`
   - `python -m mia_rl.scripts.run_blackjack_prediction`
   - `python -m mia_rl.scripts.run_blackjack_nstep_comparison`
   - `python -m mia_rl.scripts.run_kbandits`
   - `python -m mia_rl.scripts.run_car_rental_dp`
   - `python -m mia_rl.scripts.run_gridworld_policy_iteration`

## Notebooks

The repository also includes interactive notebooks in [mia_rl/notebooks/](mia_rl/notebooks/):

- [KBandits_Demo.ipynb](mia_rl/notebooks/KBandits_Demo.ipynb)
- [TicTacToe_Demo.ipynb](mia_rl/notebooks/TicTacToe_Demo.ipynb)
- [TicTacToe_MCTS.ipynb](mia_rl/notebooks/TicTacToe_MCTS.ipynb)
- [TicTacToe_PolicyGradient.ipynb](mia_rl/notebooks/TicTacToe_PolicyGradient.ipynb)
