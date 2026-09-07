"""Boundary and fail-closed contracts for public/core APIs."""

import math
from dataclasses import replace

import pytest

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs import (
    DelayedEffectEnv,
    RecoveryEnv,
    StopProbabilityPolicy,
)
from agent_credit_bench.estimators import EstimatorContext, OracleAdvantage
from agent_credit_bench.estimators.monte_carlo import MonteCarloAdvantage
from agent_credit_bench.gradients import (
    cosine_similarity,
    exact_policy_gradient,
    state_visitation,
)
from agent_credit_bench.integrations import openrlhf as openrlhf_integration
from agent_credit_bench.integrations import trl as trl_integration
from agent_credit_bench.integrations import verl as verl_integration
from agent_credit_bench.integrations.bridge import minimum_return_action
from agent_credit_bench.integrations.openrlhf import OpenRLHFGAE, OpenRLHFOutcome
from agent_credit_bench.integrations.verl import VerlGAE
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import TabularPolicy, UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Step, Trajectory, Transition
from helpers import TableMDP


@pytest.mark.parametrize("order", [("HUGE", "SAFE"), ("SAFE", "HUGE")])
def test_off_policy_reward_cannot_erase_on_policy_value(order):
    table = {
        (0, "root", "WAIT"): (Transition("s", 0.0, 1.0, False),),
        (0, "root", "QUIT"): (Transition("done", 0.5, 1.0, True),),
    }
    for action in order:
        table[1, "s", action] = (
            Transition("done", 1e16 if action == "HUGE" else 1.0, 1.0, True),
        )
    mdp = TableMDP(2, "root", table)
    policy = TabularPolicy({
        (0, "root"): {"WAIT": 0.5, "QUIT": 0.5},
        (1, "s"): {"HUGE": 0.0, "SAFE": 1.0},
    })
    values = solve_exact_values(mdp, policy)
    assert values.state_values[1, "s"] == 1.0
    assert values.advantages[0, "root", "WAIT"] == 0.25
    assert values.advantages[1, "s", "SAFE"] == 0.0
    assert exact_policy_gradient(mdp, policy, values)[0, "root", "WAIT"] == 0.125


def test_accepted_probability_roundoff_is_normalized_for_all_consumers():
    from agent_credit_bench._validation import (
        validated_policy_probabilities,
        validated_transitions,
    )

    mdp = TableMDP(2, "s", {
        (0, "s", "A"): (
            Transition("next", 1e9, 0.5, False),
            Transition("next", 1e9, 0.4999999995, False),
        ),
        (0, "s", "B"): (Transition("next", 1e9, 1.0, False),),
        (1, "next", "END"): (Transition("done", 0.0, 1.0, True),),
    })
    policy = TabularPolicy({
        (0, "s"): {"A": 0.5, "B": 0.4999999995},
        (1, "next"): {"END": 1.0},
    })
    probs = validated_policy_probabilities(policy, 0, "s", ["A", "B"])
    assert math.fsum(probs.values()) == 1.0
    transitions = validated_transitions(mdp.transitions(0, "s", "A"), 0, "s", "A")
    assert math.fsum(tr.probability for tr in transitions) == 1.0
    values = solve_exact_values(mdp, policy)
    assert values.state_values[0, "s"] == 1e9
    assert set(values.advantages.values()) == {0.0}
    assert set(exact_policy_gradient(mdp, policy, values).values()) == {0.0}
    assert state_visitation(mdp, policy)[1, "next"] == pytest.approx(1.0)
    trajectories = sample_trajectories(mdp, policy, 32, 0)
    assert {trajectory.total_return for trajectory in trajectories} == {1e9}
    credits = MonteCarloAdvantage(4).estimate(
        EstimatorContext(mdp, policy, trajectories)
    )
    assert {credit for row in credits for credit in row} == {0.0}
    # The tolerance is for roundoff, not arbitrary relative weights.
    invalid = TabularPolicy({
        (0, "s"): {"A": 0.5, "B": 0.49}, (1, "next"): {"END": 1.0}
    })
    with pytest.raises(ValueError, match="sum to"):
        solve_exact_values(mdp, invalid)


@pytest.mark.parametrize("scale", [1e-170, 1.0, 1e170])
def test_cosine_is_scale_invariant_for_finite_vectors(scale):
    from agent_credit_bench.gradients import norm

    g = {(0, "s", "A"): 3 * scale, (0, "s", "B"): 4 * scale}
    assert norm(g) == pytest.approx(5 * scale, rel=1e-15, abs=0.0)
    assert cosine_similarity(g, g) == pytest.approx(1.0)
    assert cosine_similarity(g, {k: -v for k, v in g.items()}) == pytest.approx(-1.0)
    orthogonal = {(0, "s", "A"): -4 / scale, (0, "s", "B"): 3 / scale}
    assert cosine_similarity(g, orthogonal) == pytest.approx(0.0, abs=1e-15)


def test_long_horizon_malformed_reply_uses_iterative_fallback():
    from agent_credit_bench.integrations.bridge import BridgeSession

    mdp = DelayedEffectEnv(horizon=512)
    session = BridgeSession(mdp)
    assert session.act("...").action == "BAD"
    assert len(solve_exact_values(mdp, UniformPolicy()).advantages) == 2046


def test_zero_probability_edges_do_not_require_unreachable_states() -> None:
    mdp = TableMDP(
        horizon=2,
        initial_state="s0",
        table={
            (0, "s0", "GO"): (
                Transition("live", 0.0, 1.0, False),
                Transition("unreachable", 0.0, 0.0, False),
            ),
            (0, "s0", "STOP"): (Transition("done", 0.0, 1.0, True),),
            (1, "live", "END"): (Transition("done", 1.0, 1.0, True),),
        },
    )
    policy = TabularPolicy(
        {(0, "s0"): {"GO": 0.5, "STOP": 0.5}, (1, "live"): {"END": 1.0}}
    )
    values = solve_exact_values(mdp, policy)
    assert values.state_values[(0, "s0")] == 0.5
    assert state_visitation(mdp, policy) == {(0, "s0"): 1.0, (1, "live"): 0.5}
    assert exact_policy_gradient(mdp, policy, values) == {
        (0, "s0", "GO"): 0.25,
        (0, "s0", "STOP"): -0.25,
        (1, "live", "END"): 0.0,
    }
    assert minimum_return_action(mdp, 0, "s0") == "STOP"

    stopped = TabularPolicy({(0, "s0"): {"GO": 0.0, "STOP": 1.0}})
    assert state_visitation(mdp, stopped) == {(0, "s0"): 1.0}


@pytest.mark.parametrize("offset", [0.0, 1e16, -1e16])
@pytest.mark.parametrize("probability", [0.25, 0.5, 0.75])
def test_oracle_advantages_and_gradient_are_reward_offset_invariant(
    offset, probability
):
    mdp = TableMDP(
        1, "s0",
        {
            (0, "s0", "A"): (Transition("done", offset, 1.0, True),),
            (0, "s0", "B"): (Transition("done", offset + 2.0, 1.0, True),),
        },
    )
    policy = TabularPolicy({(0, "s0"): {"A": probability, "B": 1 - probability}})
    values = solve_exact_values(mdp, policy)
    assert values.advantages == {
        (0, "s0", "A"): -2 * (1 - probability),
        (0, "s0", "B"): 2 * probability,
    }
    expected = {
        (0, "s0", "A"): -2 * probability * (1 - probability),
        (0, "s0", "B"): 2 * probability * (1 - probability),
    }
    assert exact_policy_gradient(mdp, policy, values) == expected
    # Retain the cross term even when supplied advantages have a residual shift.
    shifted = replace(
        values, advantages={key: value + 3 for key, value in values.advantages.items()}
    )
    assert exact_policy_gradient(mdp, policy, shifted) == expected
    if probability == 0.5:
        result = run_benchmark(mdp, policy, OracleAdvantage(), 16, [0, 1])
        assert result.mean_gradient_cosine == pytest.approx(1.0)


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
    assert aligned == pytest.approx(1.0)


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
