"""M2 DelayedEffectEnv tests (plan.md §12.2, §13-M2).

Representation-agnostic on purpose: they only assume the contract in the
env's module docstring ("GOOD"/"BAD" at t=0, reward 1/0 at the end), never a
particular state encoding.
"""

import pytest

from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy


def make_env(horizon: int = 4) -> DelayedEffectEnv:
    return DelayedEffectEnv(
        horizon=horizon,
        good_success_probability=0.8,
        bad_success_probability=0.2,
        num_distractor_actions=2,
    )


def every_state_action(env):
    for t in range(env.horizon):
        for s in env.states_at(t):
            for a in env.actions(t, s):
                yield t, s, a


def test_first_timestep_actions():
    env = make_env()
    assert set(env.actions(0, env.initial_state)) == {"GOOD", "BAD"}


def test_num_distractor_actions_respected():
    env = make_env()
    for t in range(1, env.horizon):
        for s in env.states_at(t):
            assert len(env.actions(t, s)) == env.num_distractor_actions


def test_transition_probabilities_sum_to_one():
    env = make_env()
    for t, s, a in every_state_action(env):
        total = sum(tr.probability for tr in env.transitions(t, s, a))
        assert total == pytest.approx(1.0, abs=1e-9)


def test_final_step_terminates():
    env = make_env()
    t = env.horizon - 1
    for s in env.states_at(t):
        for a in env.actions(t, s):
            assert all(tr.terminated for tr in env.transitions(t, s, a))


def test_nonfinal_steps_do_not_terminate():
    env = make_env()
    for t, s, a in every_state_action(env):
        if t < env.horizon - 1:
            assert all(not tr.terminated for tr in env.transitions(t, s, a))


def test_good_and_bad_q_values_are_exact():
    """With gamma=1, no intermediate reward, success pays 1: Q0 = p(success)."""
    env = make_env()
    values = solve_exact_values(env, UniformPolicy())
    s0 = env.initial_state
    assert values.action_values[(0, s0, "GOOD")] == pytest.approx(0.8)
    assert values.action_values[(0, s0, "BAD")] == pytest.approx(0.2)


@pytest.mark.parametrize("horizon", [2, 4, 8])
def test_distractors_have_equal_q_and_zero_advantage(horizon):
    env = make_env(horizon)
    values = solve_exact_values(env, UniformPolicy())
    for t in range(1, env.horizon):
        for s in env.states_at(t):
            actions = env.actions(t, s)
            q_values = [values.action_values[(t, s, a)] for a in actions]
            assert q_values == pytest.approx([q_values[0]] * len(q_values))
            for a in actions:
                assert values.advantages[(t, s, a)] == pytest.approx(0.0, abs=1e-9)
