from __future__ import annotations

from typing import TypeVar

from mia_rl.mdps.base import Action, State, TabularMDP

StateValue = dict[State, float]
DeterministicPolicy = dict[State, Action]


def one_step_q(
    mdp: TabularMDP[State, Action],
    values: StateValue,
    state: State,
    action: Action,
    gamma: float,
) -> float:
    total = 0.0
    for prob, next_state, reward, done in mdp.transitions(state, action):
        continuation = 0.0 if done else gamma * values[next_state]
        total += prob * (reward + continuation)
    return total


def policy_evaluation(
    mdp: TabularMDP[State, Action],
    policy: DeterministicPolicy,
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 10_000,
    initial_values: StateValue | None = None,
) -> tuple[StateValue, int]:
    values: StateValue = {state: 0.0 for state in mdp.states()}
    if initial_values is not None:
        values.update(initial_values)

    for it in range(max_iters):
        delta = 0.0
        old_values = values.copy()

        for state in mdp.states():
            if mdp.is_terminal(state):
                values[state] = 0.0
                continue

            action = policy[state]
            updated = one_step_q(mdp, old_values, state, action, gamma)
            delta = max(delta, abs(updated - values[state]))
            values[state] = updated

        if delta < theta:
            return values, it + 1

    return values, max_iters


def greedy_policy_improvement(
    mdp: TabularMDP[State, Action],
    values: StateValue,
    gamma: float,
    old_policy: DeterministicPolicy | None = None,
) -> tuple[DeterministicPolicy, bool]:
    new_policy: DeterministicPolicy = {}
    stable = True

    for state in mdp.states():
        if mdp.is_terminal(state):
            continue

        best_action: Action | None = None
        best_q = float("-inf")

        for action in mdp.possible_actions(state):
            q_value = one_step_q(mdp, values, state, action, gamma)
            if q_value > best_q:
                best_q = q_value
                best_action = action

        if best_action is None:
            continue

        new_policy[state] = best_action

        if old_policy is not None and old_policy.get(state) != best_action:
            stable = False

    return new_policy, stable


def _default_policy(mdp: TabularMDP[State, Action]) -> DeterministicPolicy:
    policy: DeterministicPolicy = {}
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        actions = mdp.possible_actions(state)
        if actions:
            policy[state] = actions[0]
    return policy


def policy_iteration(
    mdp: TabularMDP[State, Action],
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_outer: int = 100,
    eval_max_iters: int = 10_000,
    initial_policy: DeterministicPolicy | None = None,
) -> tuple[StateValue, DeterministicPolicy, list[tuple[int, int]]]:
    policy = _default_policy(mdp) if initial_policy is None else dict(initial_policy)
    history: list[tuple[int, int]] = []

    values: StateValue = {state: 0.0 for state in mdp.states()}

    for outer in range(max_outer):
        values, eval_iters = policy_evaluation(
            mdp=mdp,
            policy=policy,
            gamma=gamma,
            theta=theta,
            max_iters=eval_max_iters,
            initial_values=values,
        )
        policy, stable = greedy_policy_improvement(mdp, values, gamma, old_policy=policy)
        history.append((outer, eval_iters))

        if stable:
            break

    return values, policy, history


def value_iteration(
    mdp: TabularMDP[State, Action],
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> tuple[StateValue, DeterministicPolicy, int]:
    values: StateValue = {state: 0.0 for state in mdp.states()}

    for it in range(max_iters):
        delta = 0.0
        old_values = values.copy()

        for state in mdp.states():
            if mdp.is_terminal(state):
                values[state] = 0.0
                continue

            actions = mdp.possible_actions(state)
            if not actions:
                continue

            updated = max(one_step_q(mdp, old_values, state, action, gamma) for action in actions)
            delta = max(delta, abs(updated - values[state]))
            values[state] = updated

        if delta < theta:
            break

    policy, _ = greedy_policy_improvement(mdp, values, gamma)
    return values, policy, it + 1
