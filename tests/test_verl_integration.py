"""Conformance tests for the verl integration (plan.md §16).

The whole module skips when verl is not installed, so the zero-dependency
default environment stays green. Run them with the [verl] extra installed;
CI has a dedicated job for it.
"""

import importlib.util
import statistics

import pytest

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs import (
    DelayedEffectEnv,
    RecoveryEnv,
    StopProbabilityPolicy,
    VariableHorizonEnv,
)
from agent_credit_bench.estimators import BatchCenteredBroadcast, EstimatorContext
from agent_credit_bench.integrations.verl import (
    VerlGAE,
    VerlGRPO,
    VerlReinforcePlusPlus,
    VerlRLOO,
    pack_trajectories,
)
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verl") is None, reason="verl is not installed"
)


def variable_horizon_context(batch_size: int = 64, seed: int = 0) -> EstimatorContext:
    mdp = VariableHorizonEnv()
    policy = StopProbabilityPolicy(stop_probability=0.5)
    trajectories = sample_trajectories(mdp, policy, batch_size, seed)
    return EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)


def recovery_context(batch_size: int = 200, seed: int = 0) -> EstimatorContext:
    mdp = RecoveryEnv()
    policy = UniformPolicy()
    trajectories = sample_trajectories(mdp, policy, batch_size, seed)
    return EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)


def test_pack_trajectories_shapes_padding_and_group() -> None:
    context = variable_horizon_context()
    batch = pack_trajectories(context.trajectories)
    lengths = [len(t.steps) for t in context.trajectories]
    assert batch.token_level_rewards.shape == (len(lengths), max(lengths))
    assert len(set(lengths)) > 1, "variable-horizon batch should vary in length"
    for i, trajectory in enumerate(context.trajectories):
        assert batch.response_mask[i].sum().item() == len(trajectory.steps)
        row_sum = batch.token_level_rewards[i].sum().item()
        assert row_sum == pytest.approx(trajectory.total_return)
        for t in range(len(trajectory.steps), max(lengths)):
            assert batch.token_level_rewards[i, t].item() == 0.0
            assert batch.response_mask[i, t].item() == 0.0
    assert set(batch.index.tolist()) == {0}


def test_pack_trajectories_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        pack_trajectories(())


def test_verl_rloo_matches_batch_centered_exactly() -> None:
    """verl's RLOO is algebraically our leave-one-out centering."""
    context = variable_horizon_context()
    verl_credits = VerlRLOO().estimate(context)
    ours = BatchCenteredBroadcast().estimate(context)
    for verl_row, our_row in zip(verl_credits, ours, strict=True):
        for verl_credit, our_credit in zip(verl_row, our_row, strict=True):
            assert verl_credit == pytest.approx(our_credit, abs=1e-9)


def test_verl_grpo_uses_sample_std_not_population_std() -> None:
    """verl divides by Bessel-corrected std + 1e-6; document the exact formula."""
    context = variable_horizon_context()
    credits = VerlGRPO().estimate(context)
    returns = [t.total_return for t in context.trajectories]
    mean = statistics.fmean(returns)
    sample_std = statistics.stdev(returns)
    population_std = statistics.pstdev(returns)
    assert sample_std != pytest.approx(population_std)
    for row, ret in zip(credits, returns, strict=True):
        expected = (ret - mean) / (sample_std + 1e-6)
        for credit in row:
            assert credit == pytest.approx(expected, abs=1e-9)


def test_verl_dr_grpo_is_mean_centering_only() -> None:
    context = variable_horizon_context()
    credits = VerlGRPO(norm_adv_by_std=False).estimate(context)
    returns = [t.total_return for t in context.trajectories]
    mean = statistics.fmean(returns)
    for row, ret in zip(credits, returns, strict=True):
        for credit in row:
            assert credit == pytest.approx(ret - mean, abs=1e-9)


def test_verl_outcome_estimators_praise_bad_on_recovery() -> None:
    """The suite's headline failure, reproduced on verl's real code."""
    context = recovery_context()
    recovered = [
        i
        for i, trajectory in enumerate(context.trajectories)
        if len(trajectory.steps) == 2
        and trajectory.steps[0].action == "BAD"
        and trajectory.steps[1].action == "RECOVER"
        and trajectory.total_return == 1.0
    ]
    assert recovered, "expected successful BAD -> RECOVER trajectories"
    for estimator in (VerlGRPO(), VerlRLOO(), VerlReinforcePlusPlus()):
        credits = estimator.estimate(context)
        for i in recovered:
            assert credits[i][0] > 0, f"{estimator.name} should praise BAD"
            assert credits[i][1] > 0, f"{estimator.name} should praise RECOVER"


def test_verl_gae_perfect_critic_lambda_zero_distractor_credit_is_constant() -> None:
    """With lam=0 and exact values, distractor-turn TD errors are exactly zero
    (deterministic transitions, zero reward), so any credit left there after
    verl's batch whitening is one shared constant — a pure value shift."""
    mdp = DelayedEffectEnv(horizon=6)
    policy = UniformPolicy()
    trajectories = sample_trajectories(mdp, policy, 64, 0)
    context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    credits = VerlGAE(lam=0.0).estimate(context)
    distractor_credits = [
        row[t] for row in credits for t in range(1, len(row) - 1)
    ]
    first = distractor_credits[0]
    assert all(c == pytest.approx(first, abs=1e-9) for c in distractor_credits)
    assert first != pytest.approx(0.0, abs=1e-12), "whitening shifts the zeros"


def test_verl_gae_rejects_unknown_critic() -> None:
    with pytest.raises(ValueError):
        VerlGAE(critic="learned")


def test_run_benchmark_accepts_verl_estimator() -> None:
    result = run_benchmark(
        mdp=VariableHorizonEnv(),
        policy=StopProbabilityPolicy(stop_probability=0.5),
        estimator=VerlRLOO(),
        batch_size=200,
        seeds=range(2),
    )
    assert result.gradient_direction_bias > 0.99
    assert all(m.gradient_cosine > 0.9 for m in result.seed_metrics)
