"""MonteCarloAdvantage tests (plan.md §9.5).

All Monte Carlo draws are seeded, so the "statistical" assertions here are
deterministic; tolerances are generous anyway.
"""

import math
from dataclasses import replace

import pytest

from agent_credit_bench.envs import RecoveryEnv
from agent_credit_bench.estimators import EstimatorContext, MonteCarloAdvantage
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Transition
from helpers import TableMDP, stochastic_case, two_step_case


def test_none_is_a_legal_forced_action():
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", None): (Transition("done", 1.0, 1.0, True),),
            (0, "s0", "B"): (Transition("done", 0.0, 1.0, True),),
        },
    )
    policy = UniformPolicy()
    trajectories = sample_trajectories(mdp, policy, 20, seed=1)
    assert {t.steps[0].action for t in trajectories} == {None, "B"}
    credits = MonteCarloAdvantage(num_rollouts=4096, seed=31).estimate(
        EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    )
    exact = solve_exact_values(mdp, policy)
    for trajectory, row in zip(trajectories, credits, strict=True):
        action = trajectory.steps[0].action
        assert row[0] == pytest.approx(exact.advantages[(0, "s0", action)], abs=0.05)


def _mc_error(mdp, policy, num_rollouts, seed=0):
    """RMSE with deterministic but independent batch/continuation streams."""
    values = solve_exact_values(mdp, policy)
    trajectories = sample_trajectories(mdp, policy, 200, seed=seed)
    context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    mc_seed = 1_000_003 + seed * 7_919
    assert mc_seed != seed
    credits = MonteCarloAdvantage(
        num_rollouts=num_rollouts, seed=mc_seed
    ).estimate(context)
    squared = [
        (credit - values.advantages[(s.timestep, s.state, s.action)]) ** 2
        for trajectory, row in zip(trajectories, credits, strict=True)
        for s, credit in zip(trajectory.steps, row, strict=True)
    ]
    return math.sqrt(sum(squared) / len(squared))


def test_matches_oracle_with_many_rollouts():
    mdp, policy = stochastic_case()
    assert _mc_error(mdp, policy, num_rollouts=2000) < 0.15


def test_two_step_case_close_to_exact():
    mdp, policy = two_step_case()
    assert _mc_error(mdp, policy, num_rollouts=1000) < 0.1


def test_rollouts_stop_after_an_early_terminal_transition():
    assert (
        _mc_error(
            RecoveryEnv(),
            UniformPolicy(),
            num_rollouts=512,
        )
        < 0.1
    )


def test_error_shrinks_with_more_rollouts():
    """The §9.5 convergence property, coarsely: 4 rollouts vs 512."""
    mdp, policy = stochastic_case()
    assert _mc_error(mdp, policy, num_rollouts=512) < _mc_error(
        mdp, policy, num_rollouts=4
    )


def test_shapes_and_determinism():
    mdp, policy = two_step_case()
    trajectories = sample_trajectories(mdp, policy, 50, seed=3)
    context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    estimator = MonteCarloAdvantage(num_rollouts=8, seed=1)
    first = estimator.estimate(context)
    second = estimator.estimate(context)
    assert first == second  # same constructor seed, same result
    for row, trajectory in zip(first, trajectories, strict=True):
        assert len(row) == len(trajectory.steps)


def test_context_seed_varies_continuations_without_mutating_estimator():
    mdp, policy = stochastic_case()
    context = EstimatorContext(
        mdp, policy, sample_trajectories(mdp, policy, 20, seed=2)
    )
    estimator = MonteCarloAdvantage(num_rollouts=16, seed=10)
    standalone = estimator.estimate(context)
    first_context = replace(context, estimator_seed=123)
    first = estimator.estimate(first_context)
    second = estimator.estimate(replace(context, estimator_seed=456))
    assert first != second
    assert first == estimator.estimate(first_context)
    assert standalone == estimator.estimate(context)
    assert first != replace(estimator, seed=11).estimate(first_context)
