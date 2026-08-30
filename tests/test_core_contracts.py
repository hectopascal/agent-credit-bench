"""Boundary and fail-closed contracts for public/core APIs."""

import math

import pytest

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs import (
    DelayedEffectEnv,
    RecoveryEnv,
    StopProbabilityPolicy,
)
from agent_credit_bench.estimators import EstimatorContext, OracleAdvantage
from agent_credit_bench.estimators.monte_carlo import MonteCarloAdvantage
from agent_credit_bench.gradients import cosine_similarity
from agent_credit_bench.integrations import openrlhf as openrlhf_integration
from agent_credit_bench.integrations import trl as trl_integration
from agent_credit_bench.integrations import verl as verl_integration
from agent_credit_bench.integrations.openrlhf import OpenRLHFGAE, OpenRLHFOutcome
from agent_credit_bench.integrations.verl import VerlGAE
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import TabularPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Step, Trajectory, Transition
from helpers import TableMDP


def _constant_bandit() -> tuple[TableMDP, TabularPolicy]:
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "A"): (Transition("done", 1.0, 1.0, True),),
            (0, "s0", "B"): (Transition("done", 1.0, 1.0, True),),
        },
    )
    policy = TabularPolicy({(0, "s0"): {"A": 0.5, "B": 0.5}})
    return mdp, policy


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf])
def test_environment_probability_parameters_are_validated(probability: float) -> None:
    with pytest.raises(ValueError):
        DelayedEffectEnv(horizon=2, good_success_probability=probability)
    with pytest.raises(ValueError):
        RecoveryEnv(recover_success_probability=probability)
    with pytest.raises(ValueError):
        StopProbabilityPolicy(probability)


def test_probability_endpoints_are_valid() -> None:
    DelayedEffectEnv(
        horizon=2,
        good_success_probability=0.0,
        bad_success_probability=1.0,
    )
    RecoveryEnv(recover_success_probability=0.0)
    StopProbabilityPolicy(1.0)


def test_delayed_effect_requires_a_distractor_action() -> None:
    with pytest.raises(ValueError, match="num_distractor_actions"):
        DelayedEffectEnv(horizon=2, num_distractor_actions=0)


def test_oracle_rejects_negative_policy_weights_even_when_they_sum_to_one() -> None:
    mdp, _ = _constant_bandit()
    policy = TabularPolicy({(0, "s0"): {"A": -0.1, "B": 1.1}})
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        solve_exact_values(mdp, policy)


def test_oracle_rejects_probabilities_for_unavailable_actions() -> None:
    mdp, _ = _constant_bandit()
    policy = TabularPolicy(
        {(0, "s0"): {"A": 0.5, "B": 0.5, "NOT_AVAILABLE": 0.0}}
    )
    with pytest.raises(ValueError, match="unavailable actions"):
        solve_exact_values(mdp, policy)


def test_oracle_rejects_negative_transition_weights_that_sum_to_one() -> None:
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "A"): (
                Transition("x", 0.0, -0.1, True),
                Transition("y", 1.0, 1.1, True),
            )
        },
    )
    policy = TabularPolicy({(0, "s0"): {"A": 1.0}})
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        solve_exact_values(mdp, policy)


def test_sampler_validates_batch_size_and_probabilities() -> None:
    mdp, policy = _constant_bandit()
    with pytest.raises(ValueError, match="batch_size"):
        sample_trajectories(mdp, policy, batch_size=0, seed=0)

    invalid = TabularPolicy({(0, "s0"): {"A": -0.1, "B": 1.1}})
    with pytest.raises(ValueError, match="policy probabilities"):
        sample_trajectories(mdp, invalid, batch_size=1, seed=0)
    with pytest.raises(TypeError, match="seed"):
        sample_trajectories(mdp, policy, batch_size=1, seed=None)


def test_benchmark_rejects_empty_batches_and_seed_sets() -> None:
    mdp, policy = _constant_bandit()
    with pytest.raises(ValueError, match="batch_size"):
        run_benchmark(mdp, policy, OracleAdvantage(), batch_size=0, seeds=[0])
    with pytest.raises(ValueError, match="seeds"):
        run_benchmark(mdp, policy, OracleAdvantage(), batch_size=1, seeds=[])
    with pytest.raises(TypeError, match="seed"):
        run_benchmark(mdp, policy, OracleAdvantage(), batch_size=1, seeds=[None])


def test_zero_exact_gradient_alignment_metrics_are_undefined() -> None:
    mdp, policy = _constant_bandit()
    result = run_benchmark(
        mdp,
        policy,
        OracleAdvantage(),
        batch_size=16,
        seeds=[0, 1],
    )
    assert result.mean_gradient_cosine is None
    assert result.relative_mean_gradient_error is None
    assert all(metric.gradient_cosine is None for metric in result.seed_metrics)
    assert cosine_similarity({}, {}) is None
    assert cosine_similarity({(0, "s0", "A"): 1.0}, {}) is None
    aligned = cosine_similarity(
        {(0, "s0", "A"): 0.1, (0, "s0", "B"): 0.2},
        {(0, "s0", "A"): 0.1, (0, "s0", "B"): 0.2},
    )
    assert aligned == 1.0


def test_monte_carlo_requires_a_positive_rollout_count() -> None:
    with pytest.raises(ValueError, match="num_rollouts"):
        MonteCarloAdvantage(num_rollouts=0)
    with pytest.raises(TypeError, match="seed"):
        MonteCarloAdvantage(seed=None)


def test_exact_integration_critics_require_undiscounted_gamma() -> None:
    with pytest.raises(ValueError, match="undiscounted"):
        VerlGAE(critic="exact", gamma=0.9)
    with pytest.raises(ValueError, match="undiscounted"):
        OpenRLHFGAE(critic="exact", gamma=0.9)

    assert VerlGAE(critic="zero", gamma=0.9).gamma == 0.9
    assert OpenRLHFGAE(critic="zero", gamma=0.9).gamma == 0.9


def _intermediate_reward_context() -> EstimatorContext:
    trajectory = Trajectory(
        (
            Step(0, "s0", "GO", 1.0, "s1", False),
            Step(1, "s1", "END", 2.0, "done", True),
        )
    )
    return EstimatorContext(
        mdp=object(),
        policy=object(),
        trajectories=(trajectory,),
    )


@pytest.mark.parametrize(
    "estimator",
    [OpenRLHFOutcome("reinforce"), OpenRLHFGAE(critic="zero")],
)
def test_openrlhf_recursive_estimators_reject_intermediate_rewards(
    estimator,
) -> None:
    with pytest.raises(ValueError, match="scalar reward interface"):
        estimator.estimate(_intermediate_reward_context())


def test_trl_transcription_rejects_unverified_runtime_version(monkeypatch) -> None:
    monkeypatch.setattr(trl_integration, "version", lambda _package: "1.13.0")
    with pytest.raises(RuntimeError, match=r"transcribes TRL 1\.12\.0"):
        trl_integration._require_supported_trl_version()


def test_verl_adapter_rejects_unverified_runtime_version(monkeypatch) -> None:
    monkeypatch.setattr(verl_integration, "version", lambda _package: "0.10.0")
    with pytest.raises(RuntimeError, match=r"targets verl 0\.9\.0"):
        verl_integration._require_supported_verl_version()


def test_openrlhf_adapter_rejects_unverified_runtime_version(monkeypatch) -> None:
    monkeypatch.setattr(
        openrlhf_integration, "version", lambda _package: "0.12.0"
    )
    with pytest.raises(RuntimeError, match=r"targets OpenRLHF 0\.11\.0"):
        openrlhf_integration._require_supported_openrlhf_version()


def test_trl_transcription_accepts_verified_runtime_version(monkeypatch) -> None:
    monkeypatch.setattr(trl_integration, "version", lambda _package: "1.12.0")
    trl_integration._require_supported_trl_version()
