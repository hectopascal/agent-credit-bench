"""Independent regressions for the September full-audit findings."""

from itertools import permutations

import pytest

from agent_credit_bench.estimators import EstimatorContext, MonteCarloAdvantage
from agent_credit_bench.gradients import exact_policy_gradient
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Transition
from helpers import TableMDP


def chain_or_quit(rewards):
    return TableMDP(
        3,
        "root",
        {
            (0, "root", "CHAIN"): (Transition("middle", rewards[0], 1.0, False),),
            (0, "root", "QUIT"): (Transition("done", 0.0, 1.0, True),),
            (1, "middle", "GO"): (Transition("last", rewards[1], 1.0, False),),
            (2, "last", "END"): (Transition("done", rewards[2], 1.0, True),),
        },
    )


def test_monte_carlo_converges_with_cancelling_dense_rewards():
    mdp, policy = chain_or_quit((1.0, 1e16, -1e16)), UniformPolicy()
    values = solve_exact_values(mdp, policy)
    trajectories = sample_trajectories(mdp, policy, 16, 0)
    context = EstimatorContext(mdp, policy, trajectories)
    credits = MonteCarloAdvantage(num_rollouts=1024).estimate(context)
    for trajectory, row in zip(trajectories, credits, strict=True):
        step = trajectory.steps[0]
        expected = values.advantages[(0, step.state, step.action)]
        assert row[0] == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("rewards", list(permutations((1e16, 1.0, -1e16))))
def test_multiturn_oracle_matches_stable_episode_returns(rewards):
    mdp, policy = chain_or_quit(rewards), UniformPolicy()
    trajectories = sample_trajectories(mdp, policy, 16, 0)
    assert {t.total_return for t in trajectories} == {0.0, 1.0}
    # Expected return is P(CHAIN), so its two-logit softmax gradient is +/- .25.
    values = solve_exact_values(mdp, policy)
    gradient = exact_policy_gradient(mdp, policy, values)
    assert gradient[(0, "root", "CHAIN")] == 0.25
    assert gradient[(0, "root", "QUIT")] == -0.25


def test_group_baselines_are_invariant_to_a_common_return_offset():
    from agent_credit_bench.estimators import BatchCenteredBroadcast, TurnLOO
    from agent_credit_bench.types import Step, Trajectory

    mdp = TableMDP(
        1,
        "root",
        {
            (0, "root", "LOW"): (Transition("done", 1e16, 1.0, True),),
            (0, "root", "HIGH"): (Transition("done", 1e16 + 2, 1.0, True),),
        },
    )
    batch = tuple(
        Trajectory((Step(0, "root", a, r, "done", True),))
        for a, r in [("LOW", 1e16), ("HIGH", 1e16 + 2)]
    )
    context = EstimatorContext(mdp, UniformPolicy(), batch)
    for estimator in (BatchCenteredBroadcast(), TurnLOO()):
        assert estimator.estimate(context) == ((-2.0,), (2.0,))


@pytest.mark.parametrize("scale", [1e-170, 1e160])
def test_rmse_preserves_representable_error_scale(scale):
    from agent_credit_bench.metrics import rmse

    assert rmse([scale, -scale], [0.0, 0.0]) == pytest.approx(scale, rel=1e-12, abs=0.0)


def test_recovery_cli_can_report_a_batch_without_good_trajectories(
    tmp_path, monkeypatch
):
    import importlib.util
    import sys
    from pathlib import Path

    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[1] / "experiments/recovery_diagnostic.py"
    spec = importlib.util.spec_from_file_location("recovery_cli_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--batch-size",
            "2",
            "--num-seeds",
            "1",
            "--seed",
            "0",
            "--out",
            str(tmp_path / "plot.png"),
            "--csv-out",
            str(tmp_path / "metrics.csv"),
        ],
    )
    mod.main()
    assert (tmp_path / "metrics.csv").exists()


@pytest.mark.parametrize("rewards", [(1e16, 1.0, -1e16), (-1e16, -1.0, 1e16)])
def test_minimum_return_preserves_dense_reward_residual(rewards):
    import math

    from agent_credit_bench.integrations.bridge import (
        minimum_episode_return,
        minimum_return_action,
    )

    mdp = chain_or_quit(rewards)
    total = math.fsum(rewards)
    assert minimum_episode_return(mdp) == min(total, 0.0)
    assert minimum_return_action(mdp, 0, "root") == ("QUIT" if total > 0 else "CHAIN")


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_normalized_groups_preserve_common_offset(order):
    from agent_credit_bench.estimators import GiGPOStyle, GRPOStyleNormalized
    from agent_credit_bench.types import Step, Trajectory

    mdp = TableMDP(
        1,
        "root",
        {
            (0, "root", "LOW"): (Transition("done", 1e16, 1.0, True),),
            (0, "root", "HIGH"): (Transition("done", 1e16 + 2, 1.0, True),),
        },
    )
    samples = [("LOW", 1e16), ("HIGH", 1e16 + 2)]
    batch = tuple(
        Trajectory((Step(0, "root", samples[i][0], samples[i][1], "done", True),))
        for i in order
    )
    context = EstimatorContext(mdp, UniformPolicy(), batch)
    for est, weight in [(GRPOStyleNormalized(), 1), (GiGPOStyle(), 2)]:
        got = est.estimate(context)
        assert [row[0] for row in got] == pytest.approx(
            [weight * (-1 if i == 0 else 1) / 1.0001 for i in order]
        )


def test_gigpo_suffix_return_uses_stable_reward_sum():
    from agent_credit_bench.estimators import GiGPOStyle
    from agent_credit_bench.types import Step, Trajectory

    mdp = chain_or_quit((1e16, 1.0, -1e16))
    chain = Trajectory(
        (
            Step(0, "root", "CHAIN", 1e16, "middle", False),
            Step(1, "middle", "GO", 1.0, "last", False),
            Step(2, "last", "END", -1e16, "done", True),
        )
    )
    quit_ = Trajectory((Step(0, "root", "QUIT", 0.0, "done", True),))
    got = GiGPOStyle(episode_weight=0).estimate(
        EstimatorContext(mdp, UniformPolicy(), (chain, quit_))
    )
    assert got[0][0] == pytest.approx(0.5 / 0.5001)
    assert got[1][0] == pytest.approx(-0.5 / 0.5001)


def test_duplicate_seeds_are_rejected():
    from agent_credit_bench.benchmark import run_benchmark
    from agent_credit_bench.envs import RecoveryEnv
    from agent_credit_bench.estimators import OracleAdvantage

    with pytest.raises(ValueError, match="unique"):
        run_benchmark(RecoveryEnv(), UniformPolicy(), OracleAdvantage(), 4, [0, 0])


def test_monte_carlo_keeps_sample_mean_gaps_under_large_offset():
    mdp = TableMDP(
        1,
        "root",
        {
            (0, "root", "LOW"): (Transition("done", 1e16, 1.0, True),),
            (0, "root", "HIGH"): (Transition("done", 1e16 + 2, 1.0, True),),
        },
    )
    batch = sample_trajectories(mdp, UniformPolicy(), 16, 0)
    credits = MonteCarloAdvantage(num_rollouts=1024).estimate(
        EstimatorContext(mdp, UniformPolicy(), batch)
    )
    by_action = {
        tr.steps[0].action: row[0] for tr, row in zip(batch, credits, strict=True)
    }
    assert by_action["HIGH"] - by_action["LOW"] == 2.0
    assert by_action["LOW"] == pytest.approx(-1.0, abs=0.1)
    assert by_action["HIGH"] == pytest.approx(1.0, abs=0.1)


def test_oracle_rejects_unrepresentable_public_values():
    mdp = chain_or_quit((1e308, 1e308, 0.0))
    with pytest.raises(ValueError, match="finite float range"):
        solve_exact_values(mdp, UniformPolicy())
