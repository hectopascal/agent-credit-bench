"""Conformance tests for the TRL integration.

The whole module skips when trl is not installed, so the zero-dependency
default environment stays green. CI runs it in a dedicated job.

The fingerprint tests are the load-bearing ones: integrations/trl.py is a
transcription of trainer-inline code, and these assert the transcribed
expressions still appear verbatim in the installed TRL source. If a TRL
release rewrites its advantage math, they fail — re-verify the transcription
before trusting any TRL numbers.
"""

import importlib.util
import inspect
import statistics

import pytest

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs import (
    RecoveryEnv,
    StopProbabilityPolicy,
    VariableHorizonEnv,
)
from agent_credit_bench.estimators import BatchCenteredBroadcast, EstimatorContext
from agent_credit_bench.integrations.trl import TrlGRPO, TrlRLOO
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("trl") is None, reason="trl is not installed"
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


def test_grpo_fingerprint_matches_installed_source() -> None:
    from trl import GRPOTrainer

    source = inspect.getsource(GRPOTrainer._generate_and_score_completions)
    for expression in (
        "mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations)",
        "std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)",
        "std_rewards = nanstd(rewards).expand_as(rewards)",
        "advantages = rewards - mean_grouped_rewards",
        "advantages = advantages / (std_rewards + 1e-4)",
    ):
        assert expression in source, (
            f"TRL changed its GRPO advantage code: {expression}"
        )


def test_rloo_fingerprint_matches_installed_source() -> None:
    from trl import RLOOTrainer

    source = inspect.getsource(RLOOTrainer._generate_and_score_completions)
    for expression in (
        "baselines = (grouped_sum - grouped_rewards) / (scorable_counts - 1)",
        "advantages = rewards - baselines",
        "advantages = (advantages - torch.nanmean(advantages))"
        " / (nanstd(advantages) + 1e-4)",
    ):
        assert expression in source, (
            f"TRL changed its RLOO advantage code: {expression}"
        )


def test_trl_rloo_matches_batch_centered_exactly() -> None:
    """TRL's default RLOO is plain leave-one-out centering."""
    context = variable_horizon_context()
    trl_credits = TrlRLOO().estimate(context)
    ours = BatchCenteredBroadcast().estimate(context)
    for trl_row, our_row in zip(trl_credits, ours, strict=True):
        for trl_credit, our_credit in zip(trl_row, our_row, strict=True):
            assert trl_credit == pytest.approx(our_credit, abs=1e-9)


def test_trl_grpo_uses_sample_std_and_1e4_epsilon() -> None:
    """TRL's nanstd is Bessel-corrected like verl's torch.std, but eps is 1e-4."""
    context = variable_horizon_context()
    credits = TrlGRPO().estimate(context)
    returns = [t.total_return for t in context.trajectories]
    mean = statistics.fmean(returns)
    sample_std = statistics.stdev(returns)
    for row, ret in zip(credits, returns, strict=True):
        expected = (ret - mean) / (sample_std + 1e-4)
        for credit in row:
            # 1e-6 abs: TRL's nanstd computes its Bessel correction in
            # float32 (int/int tensor division), so float64 input still
            # carries ~1e-8 error. Population-vs-sample std or a different
            # epsilon would miss by far more than this tolerance.
            assert credit == pytest.approx(expected, abs=1e-6)


def test_trl_dr_grpo_is_mean_centering_only() -> None:
    context = variable_horizon_context()
    credits = TrlGRPO(scale_rewards="none").estimate(context)
    returns = [t.total_return for t in context.trajectories]
    mean = statistics.fmean(returns)
    for row, ret in zip(credits, returns, strict=True):
        for credit in row:
            assert credit == pytest.approx(ret - mean, abs=1e-9)


def test_trl_batch_scale_equals_group_scale_for_single_group() -> None:
    context = variable_horizon_context()
    group = TrlGRPO(scale_rewards="group").estimate(context)
    batch = TrlGRPO(scale_rewards="batch").estimate(context)
    for group_row, batch_row in zip(group, batch, strict=True):
        for g, b in zip(group_row, batch_row, strict=True):
            assert g == pytest.approx(b, abs=1e-12)


def test_trl_grpo_rejects_unknown_scale() -> None:
    with pytest.raises(ValueError):
        TrlGRPO(scale_rewards="global")


def test_trl_rloo_rejects_single_trajectory() -> None:
    context = recovery_context(batch_size=1)
    with pytest.raises(ValueError):
        TrlRLOO().estimate(context)


def test_trl_outcome_estimators_are_positive_on_selected_recovery() -> None:
    """Check the conditional successful-repair credit diagnostic."""
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
    for estimator in (TrlGRPO(), TrlRLOO(), TrlRLOO(normalize_advantages=True)):
        credits = estimator.estimate(context)
        for i in recovered:
            assert credits[i][0] > 0, f"{estimator.name}: expected BAD > 0"
            assert credits[i][1] > 0, f"{estimator.name}: expected RECOVER > 0"


def test_run_benchmark_accepts_trl_estimator() -> None:
    result = run_benchmark(
        mdp=VariableHorizonEnv(),
        policy=StopProbabilityPolicy(stop_probability=0.5),
        estimator=TrlRLOO(),
        batch_size=200,
        seeds=range(2),
    )
    assert result.mean_gradient_cosine > 0.99
    assert all(m.gradient_cosine > 0.9 for m in result.seed_metrics)
