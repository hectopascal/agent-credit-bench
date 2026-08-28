"""MonteCarloAdvantage tests (plan.md §9.5).

All Monte Carlo draws are seeded, so the "statistical" assertions here are
deterministic; tolerances are generous anyway.
"""

import math

from agent_credit_bench.estimators import EstimatorContext, MonteCarloAdvantage
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.sampling import sample_trajectories
from helpers import stochastic_case, two_step_case


def _mc_error(mdp, policy, num_rollouts, seed=0):
    """RMSE of MC credit vs exact advantage over a sampled batch."""
    values = solve_exact_values(mdp, policy)
    trajectories = sample_trajectories(mdp, policy, 200, seed=seed)
    context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    credits = MonteCarloAdvantage(num_rollouts=num_rollouts, seed=seed).estimate(
        context
    )
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
