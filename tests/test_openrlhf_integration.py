"""Conformance tests for the OpenRLHF integration.

The whole module skips when openrlhf is not installed. OpenRLHF ships
Linux-x86_64-only wheels, so on macOS these always skip — CI runs them in a
dedicated Linux job, and they can be run locally via Docker (see
docs/openrlhf_integration.md).

The pipeline's whitening uses population variance (unlike verl's
Bessel-corrected helper) and computes its mean/rstd in float32. Tests compare
against the population formula with float32-aware tolerances; unwhitened
estimators (rloo, group_norm, dr_grpo) are asserted at 1e-9.
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
from agent_credit_bench.integrations.openrlhf import OpenRLHFGAE, OpenRLHFOutcome
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openrlhf") is None, reason="openrlhf is not installed"
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


def test_openrlhf_rloo_matches_batch_centered_exactly() -> None:
    """OpenRLHF's rloo is leave-one-out centering, like verl's and TRL's."""
    context = variable_horizon_context()
    openrlhf_credits = OpenRLHFOutcome("rloo").estimate(context)
    ours = BatchCenteredBroadcast().estimate(context)
    for openrlhf_row, our_row in zip(openrlhf_credits, ours, strict=True):
        for openrlhf_credit, our_credit in zip(openrlhf_row, our_row, strict=True):
            assert openrlhf_credit == pytest.approx(our_credit, abs=1e-9)


def test_openrlhf_group_norm_formula() -> None:
    """Their GRPO: Bessel-corrected sample std, epsilon 1e-9 (verl: 1e-6, TRL: 1e-4)."""
    context = variable_horizon_context()
    credits = OpenRLHFOutcome("group_norm").estimate(context)
    returns = [t.total_return for t in context.trajectories]
    mean = statistics.fmean(returns)
    sample_std = statistics.stdev(returns)
    for row, ret in zip(credits, returns, strict=True):
        expected = (ret - mean) / (sample_std + 1e-9)
        for credit in row:
            assert credit == pytest.approx(expected, abs=1e-9)


def test_openrlhf_dr_grpo_is_mean_centering_only() -> None:
    context = variable_horizon_context()
    credits = OpenRLHFOutcome("dr_grpo").estimate(context)
    returns = [t.total_return for t in context.trajectories]
    mean = statistics.fmean(returns)
    for row, ret in zip(credits, returns, strict=True):
        for credit in row:
            assert credit == pytest.approx(ret - mean, abs=1e-9)


def test_openrlhf_reinforce_baseline_is_whitened_dr_grpo() -> None:
    """The whitening pass uses population, not Bessel-corrected, variance."""
    context = variable_horizon_context()
    whitened = OpenRLHFOutcome("reinforce_baseline").estimate(context)
    centered = OpenRLHFOutcome("dr_grpo").estimate(context)
    flat_whitened = [c for row in whitened for c in row]
    flat_centered = [c for row in centered for c in row]
    assert statistics.fmean(flat_whitened) == pytest.approx(0.0, abs=1e-5)
    assert statistics.pstdev(flat_whitened) == pytest.approx(1.0, abs=1e-4)
    scale = statistics.pstdev(flat_centered)
    sample_scale = statistics.stdev(flat_centered)
    assert scale != pytest.approx(sample_scale)
    shift = statistics.fmean(flat_centered)
    for w, c in zip(flat_whitened, flat_centered, strict=True):
        assert w == pytest.approx((c - shift) / scale, abs=1e-4)


def test_openrlhf_outcome_estimators_are_positive_on_selected_recovery() -> None:
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
    for estimator_name in ("rloo", "group_norm", "reinforce", "reinforce_baseline"):
        credits = OpenRLHFOutcome(estimator_name).estimate(context)
        for i in recovered:
            assert credits[i][0] > 0, f"{estimator_name}: expected BAD > 0"
            assert credits[i][1] > 0, f"{estimator_name}: expected RECOVER > 0"


def test_openrlhf_gae_default_lambda_is_positive_on_selected_bad() -> None:
    """OpenRLHF's default lambd=1 has the same blind spot as verl's."""
    context = recovery_context()
    recovered = [
        i
        for i, trajectory in enumerate(context.trajectories)
        if len(trajectory.steps) == 2
        and trajectory.steps[0].action == "BAD"
        and trajectory.total_return == 1.0
    ]
    lam1 = OpenRLHFGAE(critic="exact", lam=1.0).estimate(context)
    lam0 = OpenRLHFGAE(critic="exact", lam=0.0).estimate(context)
    for i in recovered:
        assert lam1[i][0] > 0, "lam=1: expected selected BAD credit > 0"
        assert lam0[i][0] < lam0[i][1], "lam=0: critic separates BAD from RECOVER"


def test_openrlhf_gae_lambda_zero_distractor_credit_is_constant() -> None:
    """Zero TD errors on distractor turns leave one shared whitening constant."""
    mdp = DelayedEffectEnv(horizon=6)
    policy = UniformPolicy()
    trajectories = sample_trajectories(mdp, policy, 64, 0)
    context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    credits = OpenRLHFGAE(critic="exact", lam=0.0).estimate(context)
    distractor_credits = [row[t] for row in credits for t in range(1, len(row) - 1)]
    first = distractor_credits[0]
    assert all(c == pytest.approx(first, abs=1e-5) for c in distractor_credits)


def test_openrlhf_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        OpenRLHFOutcome("gae")  # gae goes through OpenRLHFGAE
    with pytest.raises(ValueError):
        OpenRLHFOutcome("rloo", gamma=0.9)  # pipeline would force it back to 1
    with pytest.raises(ValueError):
        OpenRLHFGAE(critic="learned")
    with pytest.raises(ValueError):
        OpenRLHFOutcome("rloo").estimate(recovery_context(batch_size=1))


def test_run_benchmark_accepts_openrlhf_estimator() -> None:
    result = run_benchmark(
        mdp=VariableHorizonEnv(),
        policy=StopProbabilityPolicy(stop_probability=0.5),
        estimator=OpenRLHFOutcome("rloo"),
        batch_size=200,
        seeds=range(2),
    )
    assert result.mean_gradient_cosine > 0.99
    assert all(m.gradient_cosine > 0.9 for m in result.seed_metrics)
