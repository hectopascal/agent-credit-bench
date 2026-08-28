"""M4 RecoveryEnv tests (plan.md §12.2, §13-M4)."""

import pytest

from agent_credit_bench.envs.recovery import RecoveryEnv
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy


def test_transition_probabilities_sum_to_one():
    env = RecoveryEnv(recover_success_probability=0.7)
    for t in range(env.horizon):
        for s in env.states_at(t):
            for a in env.actions(t, s):
                total = sum(tr.probability for tr in env.transitions(t, s, a))
                assert total == pytest.approx(1.0, abs=1e-9)


def test_recover_beats_give_up():
    env = RecoveryEnv(recover_success_probability=0.7)
    values = solve_exact_values(env, UniformPolicy())
    assert (
        values.action_values[(1, "mistake", "RECOVER")]
        > values.action_values[(1, "mistake", "GIVE_UP")]
    )


@pytest.mark.parametrize("q", [0.5, 1.0])
def test_oracle_sign_pattern_under_uniform_policy(q):
    """BAD negative, RECOVER positive — the pattern the diagnostic relies on."""
    env = RecoveryEnv(recover_success_probability=q)
    values = solve_exact_values(env, UniformPolicy())
    assert values.advantages[(0, "s0", "BAD")] < 0
    assert values.advantages[(1, "mistake", "RECOVER")] > 0


def test_hand_calculated_values_for_certain_recovery():
    """The q=1 numbers from the module docstring."""
    env = RecoveryEnv(recover_success_probability=1.0)
    values = solve_exact_values(env, UniformPolicy())
    assert values.action_values[(0, "s0", "BAD")] == pytest.approx(0.5)
    assert values.state_values[(0, "s0")] == pytest.approx(0.75)
    assert values.advantages[(0, "s0", "BAD")] == pytest.approx(-0.25)
    assert values.advantages[(1, "mistake", "RECOVER")] == pytest.approx(0.5)
