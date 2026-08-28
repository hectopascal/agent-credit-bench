"""M2 sampler tests (plan.md §12.3, §13-M2).

Statistical tests use generous tolerances on purpose — flaky tests are worse
than loose ones.
"""

import pytest

from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import TabularPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Transition
from helpers import TableMDP, bandit_case, stochastic_case


def chain_case() -> tuple[TableMDP, TabularPolicy]:
    """Three non-terminating steps; only the horizon ends the episode."""
    mdp = TableMDP(
        horizon=3,
        initial_state="a",
        table={
            (0, "a", "NEXT"): (Transition("b", 0.0, 1.0, False),),
            (1, "b", "NEXT"): (Transition("c", 0.0, 1.0, False),),
            (2, "c", "NEXT"): (Transition("d", 1.0, 1.0, False),),
        },
    )
    policy = TabularPolicy(
        {
            (0, "a"): {"NEXT": 1.0},
            (1, "b"): {"NEXT": 1.0},
            (2, "c"): {"NEXT": 1.0},
        }
    )
    return mdp, policy


def test_step_alignment_and_horizon():
    mdp, policy = chain_case()
    trajectories = sample_trajectories(mdp, policy, batch_size=8, seed=0)

    assert len(trajectories) == 8
    for trajectory in trajectories:
        steps = trajectory.steps
        # Non-terminating chain: the horizon must end the episode.
        assert len(steps) == mdp.horizon
        assert steps[0].state == mdp.initial_state
        for i, step in enumerate(steps):
            assert step.timestep == i
        for earlier, later in zip(steps, steps[1:], strict=False):
            assert not earlier.terminated
            assert earlier.next_state == later.state


def test_termination_stops_sampling():
    mdp, policy = bandit_case()  # horizon 1, every transition terminates
    trajectories = sample_trajectories(mdp, policy, batch_size=8, seed=0)
    for trajectory in trajectories:
        assert len(trajectory.steps) == 1
        assert trajectory.steps[-1].terminated


def test_seeded_sampling_is_reproducible():
    mdp, policy = stochastic_case()
    first = sample_trajectories(mdp, policy, batch_size=32, seed=7)
    second = sample_trajectories(mdp, policy, batch_size=32, seed=7)
    other = sample_trajectories(mdp, policy, batch_size=32, seed=8)

    assert first == second
    assert first != other  # astronomically unlikely to collide


def test_action_frequencies_approach_policy():
    mdp, policy = bandit_case()
    trajectories = sample_trajectories(mdp, policy, batch_size=4000, seed=0)
    freq_a = sum(t.steps[0].action == "A" for t in trajectories) / len(trajectories)
    assert freq_a == pytest.approx(0.5, abs=0.05)


def test_transition_frequencies_approach_probabilities():
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "GAMBLE"): (
                Transition("win", 10.0, 0.3, True),
                Transition("lose", 0.0, 0.7, True),
            ),
        },
    )
    policy = TabularPolicy({(0, "s0"): {"GAMBLE": 1.0}})
    trajectories = sample_trajectories(mdp, policy, batch_size=4000, seed=0)
    freq_win = sum(t.steps[0].next_state == "win" for t in trajectories) / len(
        trajectories
    )
    assert freq_win == pytest.approx(0.3, abs=0.05)


def test_mean_return_approaches_initial_state_value():
    """Integration check: the sampler agrees with the M1 oracle."""
    mdp, policy = stochastic_case()
    values = solve_exact_values(mdp, policy)
    trajectories = sample_trajectories(mdp, policy, batch_size=4000, seed=0)
    mean_return = sum(t.total_return for t in trajectories) / len(trajectories)
    assert mean_return == pytest.approx(values.state_values[(0, "s0")], abs=0.3)
