"""Tests for the adapter and the library-style baselines (plan.md §16)."""

import statistics

import pytest

from agent_credit_bench.envs import (
    DelayedEffectEnv,
    StopProbabilityPolicy,
    VariableHorizonEnv,
)
from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    GiGPOStyle,
    GRPOStyleNormalized,
    TrajectoryReturnAdapter,
)
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Step, Trajectory
from helpers import bandit_case


def bandit_batch() -> EstimatorContext:
    mdp, policy = bandit_case()
    trajectories = (
        Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),
        Trajectory((Step(0, "s0", "B", 0.0, "done", True),)),
        Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),
    )
    return EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)


def test_adapter_reproduces_batch_centered():
    """A hand-written LOO function through the adapter == the built-in."""

    def loo(returns):
        total = sum(returns)
        n = len(returns)
        return [r - (total - r) / (n - 1) for r in returns]

    context = bandit_batch()
    adapted = TrajectoryReturnAdapter(fn=loo, name="loo_adapter")
    expected = BatchCenteredBroadcast().estimate(context)
    for adapted_row, expected_row in zip(
        adapted.estimate(context), expected, strict=True
    ):
        assert adapted_row == pytest.approx(expected_row)


def test_adapter_rejects_wrong_output_length():
    adapter = TrajectoryReturnAdapter(fn=lambda returns: returns[:-1])
    with pytest.raises(ValueError):
        adapter.estimate(bandit_batch())


def test_grpo_style_hand_calculation():
    # Returns 1, 0, 1: mean 2/3, population std sqrt(2)/3.
    context = bandit_batch()
    credits = GRPOStyleNormalized().estimate(context)
    std = statistics.pstdev([1.0, 0.0, 1.0])
    assert credits[0][0] == pytest.approx((1 / 3) / (std + 1e-4))
    assert credits[1][0] == pytest.approx((-2 / 3) / (std + 1e-4))
    assert credits[2][0] == pytest.approx(credits[0][0])


def test_grpo_style_constant_returns_give_zero_credit():
    mdp, policy = bandit_case()
    context = EstimatorContext(
        mdp=mdp,
        policy=policy,
        trajectories=(
            Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),
            Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),
        ),
    )
    credits = GRPOStyleNormalized().estimate(context)
    assert credits == ((0.0,), (0.0,))


def test_gigpo_episode_level_term_and_population_std_hand_calculation():
    # Returns 1, 0, 1: the episode term must match the population-normalized
    # GRPO term, and the default estimator must add the identical step term.
    context = bandit_batch()
    episode_only = GiGPOStyle(step_weight=0.0).estimate(context)
    combined = GiGPOStyle().estimate(context)
    returns = [1.0, 0.0, 1.0]
    mean = statistics.fmean(returns)
    population_std = statistics.pstdev(returns)
    expected = [(value - mean) / (population_std + 1e-4) for value in returns]

    assert population_std != pytest.approx(statistics.stdev(returns))
    for index, z_score in enumerate(expected):
        assert episode_only[index][0] == pytest.approx(z_score)
        assert combined[index][0] == pytest.approx(2.0 * z_score)


def _variable_horizon_gigpo_context(
    continue_reward: float,
) -> EstimatorContext:
    env = VariableHorizonEnv(
        stop_rewards=(0.0, 2.0, 1.0),
        continue_reward=continue_reward,
    )
    trajectories = (
        Trajectory((Step(0, "alive", "STOP", 0.0, "done", True),)),
        Trajectory(
            (
                Step(0, "alive", "CONTINUE", continue_reward, "alive", False),
                Step(1, "alive", "STOP", 2.0, "done", True),
            )
        ),
        Trajectory(
            (
                Step(0, "alive", "CONTINUE", continue_reward, "alive", False),
                Step(1, "alive", "CONTINUE", continue_reward, "alive", False),
                Step(2, "alive", "STOP", 1.0, "done", True),
            )
        ),
    )
    return EstimatorContext(
        mdp=env,
        policy=StopProbabilityPolicy(0.5),
        trajectories=trajectories,
    )


def test_gigpo_groups_recurrent_anchor_state_across_timesteps():
    context = _variable_horizon_gigpo_context(continue_reward=0.0)
    credits = GiGPOStyle(episode_weight=0.0).estimate(context)
    pooled_returns_to_go = [0.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    mean = statistics.fmean(pooled_returns_to_go)
    std = statistics.pstdev(pooled_returns_to_go)
    expected = (1.0 - mean) / (std + 1e-4)

    # The same "alive" anchor and RTG=1 recur at t=0, 1, and 2. GiGPO pools
    # them by state; grouping by (t, state) would make the t=2 singleton zero.
    assert credits[2] == pytest.approx((expected, expected, expected))
    assert expected != pytest.approx(0.0)


def test_gigpo_uses_return_to_go_with_dense_rewards():
    context = _variable_horizon_gigpo_context(continue_reward=1.0)
    credits = GiGPOStyle(episode_weight=0.0).estimate(context)
    pooled_returns_to_go = [0.0, 3.0, 2.0, 3.0, 2.0, 1.0]
    mean = statistics.fmean(pooled_returns_to_go)
    std = statistics.pstdev(pooled_returns_to_go)

    def z_score(value: float) -> float:
        return (value - mean) / (std + 1e-4)

    # This trajectory's full return is 3 at both turns, while its true
    # returns-to-go are 3 then 2. Replacing RTG with episode return fails.
    assert credits[1] == pytest.approx((z_score(3.0), z_score(2.0)))
    assert credits[1][0] != pytest.approx(credits[1][1])


def test_gigpo_step_level_separates_latent_states():
    """Anchor-state grouping should give distractor steps state-conditioned
    credit: in the delayed-effect env, a "good"-state step and a "bad"-state
    step see different return-to-go groups."""
    env = DelayedEffectEnv(horizon=4)
    policy = UniformPolicy()
    trajectories = sample_trajectories(env, policy, 500, seed=0)
    context = EstimatorContext(mdp=env, policy=policy, trajectories=trajectories)
    credits = GiGPOStyle(episode_weight=0.0, step_weight=1.0).estimate(context)

    # Step-level-only credit at t>=1 must depend only on (state, return-to-go);
    # two successful trajectories in the same latent state get equal credit.
    by_key = {}
    for trajectory, row in zip(trajectories, credits, strict=True):
        for step, credit in zip(trajectory.steps, row, strict=True):
            if step.timestep >= 1:
                key = (step.state, trajectory.total_return)
                by_key.setdefault(key, set()).add(round(credit, 9))
    assert all(len(values) == 1 for values in by_key.values())
    # And "good" vs "bad" groups are actually distinguished for successes.
    good = next(iter(by_key[("good", 1.0)]))
    bad = next(iter(by_key[("bad", 1.0)]))
    assert good != bad


def test_gigpo_shapes():
    context = bandit_batch()
    credits = GiGPOStyle().estimate(context)
    assert len(credits) == 3
    assert all(len(row) == 1 for row in credits)
