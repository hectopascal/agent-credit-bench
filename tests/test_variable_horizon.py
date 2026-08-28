"""M5 VariableHorizonEnv and TurnLOO tests (plan.md §12.2, §12.4, §13-M5)."""

import pytest

from agent_credit_bench.envs.variable_horizon import (
    StopProbabilityPolicy,
    VariableHorizonEnv,
)
from agent_credit_bench.estimators import EstimatorContext
from agent_credit_bench.estimators.turn_loo import TurnLOO
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.types import Step, Trajectory


def test_structure():
    env = VariableHorizonEnv()
    assert env.horizon == 5
    for t in range(env.horizon):
        for s in env.states_at(t):
            expected = ["STOP"] if t == env.horizon - 1 else ["STOP", "CONTINUE"]
            assert env.actions(t, s) == expected
            for a in env.actions(t, s):
                total = sum(tr.probability for tr in env.transitions(t, s, a))
                assert total == pytest.approx(1.0)
    # The final timestep forces termination.
    for tr in env.transitions(env.horizon - 1, "alive", "STOP"):
        assert tr.terminated


def test_continue_advantage_changes_sign_across_timesteps():
    """Hand calculation for stop_rewards (0,2,1,3,0) at stop probability 0.5.

    Backwards: V4=0; Q(CONT,3)=0 vs stop 3 -> negative; V3=1.5;
    Q(CONT,2)=1.5 vs stop 1 -> positive; V2=1.25; Q(CONT,1)=1.25 vs stop 2
    -> negative; V1=1.625; Q(CONT,0)=1.625 vs stop 0 -> positive; V0=0.8125.
    """
    env = VariableHorizonEnv()
    values = solve_exact_values(env, StopProbabilityPolicy(0.5))

    assert values.advantages[(0, "alive", "CONTINUE")] == pytest.approx(0.8125)
    assert values.advantages[(1, "alive", "CONTINUE")] < 0
    assert values.advantages[(2, "alive", "CONTINUE")] > 0
    assert values.advantages[(3, "alive", "CONTINUE")] == pytest.approx(-1.5)


def _trajectory(actions_and_rewards):
    steps = []
    for t, (action, reward, terminated) in enumerate(actions_and_rewards):
        next_state = "done" if terminated else "alive"
        steps.append(Step(t, "alive", action, reward, next_state, terminated))
    return Trajectory(tuple(steps))


def turn_loo_context():
    env = VariableHorizonEnv()
    policy = StopProbabilityPolicy(0.5)
    trajectories = (
        _trajectory([("STOP", 0.0, True)]),  # return 0, length 1
        _trajectory([("CONTINUE", 0.0, False), ("STOP", 2.0, True)]),  # return 2
        _trajectory(
            [
                ("CONTINUE", 0.0, False),
                ("CONTINUE", 0.0, False),
                ("CONTINUE", 0.0, False),
                ("STOP", 3.0, True),
            ]
        ),  # return 3, length 4
    )
    return EstimatorContext(mdp=env, policy=policy, trajectories=trajectories)


def test_turn_loo_hand_calculation():
    credits = TurnLOO().estimate(turn_loo_context())

    # t=0: all three active. LOO baselines: mean(2,3), mean(0,3), mean(0,2).
    assert credits[0] == pytest.approx((-2.5,))
    assert credits[1][0] == pytest.approx(0.5)
    assert credits[2][0] == pytest.approx(2.0)
    # t=1: only trajectories 1 and 2 active.
    assert credits[1][1] == pytest.approx(-1.0)
    assert credits[2][1] == pytest.approx(1.0)
    # t=2, t=3: trajectory 2 is alone — no peers, credit defined as 0.0.
    assert credits[2][2] == 0.0
    assert credits[2][3] == 0.0


def test_turn_loo_shapes():
    context = turn_loo_context()
    credits = TurnLOO().estimate(context)
    assert len(credits) == len(context.trajectories)
    for row, trajectory in zip(credits, context.trajectories, strict=True):
        assert len(row) == len(trajectory.steps)
