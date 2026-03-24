# `mia_rl`

## Package organization

- `mia_rl/core/`
  - generic abstractions such as `Environment`, `Policy`, `Agent`, `Episode`, `Transition`
- `mia_rl/envs/`
  - interactive environments for model-free methods
- `mia_rl/mdps/`
  - known-model MDP abstractions for dynamic programming classes
- `mia_rl/policies/`
  - reusable policies
- `mia_rl/agents/`
  - learning algorithms grouped by task
- `mia_rl/plots/`
  - reusable plotting functions
- `mia_rl/experiments/`
  - rollout, training, evaluation, and experiment helpers
- `mia_rl/scripts/`
  - runnable experiment scripts
- `mia_rl/outputs/`
  - generated plots and experiment results

## Current practical TODOs

For the Windy Gridworld Sarsa practical, complete:
- `mia_rl/envs/windy_gridworld.py`
- `mia_rl/agents/control/sarsa.py`


Optional TODO extension after the main practical is complete (n-step Sarsa):
- `mia_rl/agents/control/n_step_sarsa.py`

## Setting up the Environment

First, create the conda environment:

- `conda env create -f environment.yml`

After implementing the TODOs, run:

- `python -m mia_rl.scripts.run_windy_gridworld_sarsa`
- `python -m mia_rl.scripts.run_windy_gridworld_n_step_sarsa`

By default, generated figures are saved under `mia_rl/outputs/`.

