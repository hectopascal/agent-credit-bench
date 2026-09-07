"""M3 benchmark-runner tests (plan.md §13-M3).

Sampling is seeded, so every assertion here is deterministic.
"""

import math

import pytest

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    MonteCarloAdvantage,
    OracleAdvantage,
    OutcomeBroadcast,
)
from agent_credit_bench.policy import UniformPolicy
from helpers import stochastic_case


def run(estimator):
    return run_benchmark(
        mdp=DelayedEffectEnv(horizon=4),
        policy=UniformPolicy(),
        estimator=estimator,
        batch_size=500,
        seeds=range(3),
    )


def test_oracle_estimator_scores_perfectly():
    result = run(OracleAdvantage())
    for m in result.seed_metrics:
        assert m.rmse == pytest.approx(0.0, abs=1e-9)
        assert m.centered_rmse == pytest.approx(0.0, abs=1e-9)
        assert m.leakage_ratio == pytest.approx(0.0, abs=1e-6)
        assert m.sign_accuracy == pytest.approx(1.0)
    assert result.mean_gradient_cosine > 0.999
    assert result.relative_mean_gradient_error < 0.2


def test_outcome_broadcast_smears_but_points_roughly_right():
    result = run(OutcomeBroadcast())
    for m in result.seed_metrics:
        assert m.leakage_ratio > 0.5  # most credit lands on zero-advantage steps
        assert m.rmse > 0.3
    # Direction agreement should be far stronger than the value error suggests.
    assert result.mean_gradient_cosine > 0.9
    # And its gradient variance should dwarf the oracle's.
    oracle = run(OracleAdvantage())
    assert result.gradient_variance > 5 * oracle.gradient_variance


def test_batch_centered_runs_and_reports_all_rows():
    result = run(BatchCenteredBroadcast())
    rows = result.rows()
    assert len(rows) == 3
    expected_keys = {
        "estimator",
        "seed",
        "estimator_seed",
        "rmse",
        "centered_rmse",
        "multi_visit_fraction",
        "spearman",
        "sign_accuracy",
        "sign_num_scored",
        "sign_num_excluded",
        "leakage_ratio",
        "gradient_cosine",
        "mean_gradient_cosine",
        "relative_mean_gradient_error",
        "gradient_direction_bias",
        "gradient_magnitude_error",
        "gradient_variance",
    }
    assert set(rows[0]) == expected_keys
    assert all(row["estimator"] == "batch_centered_broadcast" for row in rows)
    assert rows[0]["gradient_direction_bias"] == rows[0]["mean_gradient_cosine"]
    assert (
        rows[0]["gradient_magnitude_error"]
        == rows[0]["relative_mean_gradient_error"]
    )


def test_per_turn_stats_cover_every_timestep():
    result = run(OracleAdvantage())
    assert set(result.per_turn) == {0, 1, 2, 3}
    # Distractor steps: oracle credit is exactly zero there.
    for t in (1, 2, 3):
        assert result.per_turn[t].bias == pytest.approx(0.0, abs=1e-9)
        assert result.per_turn[t].variance == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("credit", [math.nan, math.inf, -math.inf])
def test_benchmark_rejects_nonfinite_estimator_credits(credit):
    class InvalidEstimator:
        name = "invalid_credit"

        def estimate(self, context):
            return tuple(
                tuple(credit for _ in trajectory.steps)
                for trajectory in context.trajectories
            )

    with pytest.raises(ValueError, match="invalid_credit.*finite.*trajectory 0"):
        run(InvalidEstimator())


def test_benchmark_resamples_continuations_across_seeds():
    mdp, policy = stochastic_case()
    result = run_benchmark(
        mdp, policy, MonteCarloAdvantage(num_rollouts=64), 16, range(2000)
    )
    assert result.relative_mean_gradient_error < 0.1


def test_benchmark_estimator_seeds_are_reproducible_and_order_independent():
    mdp, policy = stochastic_case()
    estimator = MonteCarloAdvantage(num_rollouts=8, seed=19)
    forward = run_benchmark(mdp, policy, estimator, 16, [0, 1, 2])
    repeated = run_benchmark(mdp, policy, estimator, 16, [0, 1, 2])
    reverse = run_benchmark(mdp, policy, estimator, 16, [2, 1, 0])
    assert forward == repeated
    assert forward.seed_metrics == tuple(reversed(reverse.seed_metrics))
    estimator_seeds = [metric.estimator_seed for metric in forward.seed_metrics]
    assert len(set(estimator_seeds)) == 3
    assert all(seed is not None and seed not in (0, 1, 2) for seed in estimator_seeds)
    assert [row["estimator_seed"] for row in forward.rows()] == estimator_seeds
