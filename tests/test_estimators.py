"""M2 estimator tests (plan.md §12.4, §13-M2).

Built on hand-made trajectories, not the sampler, so estimator bugs and
sampler bugs fail different tests.
"""

import math

import pytest

from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    OracleAdvantage,
    OutcomeBroadcast,
)
from agent_credit_bench.types import Step, Trajectory
from helpers import bandit_case, two_step_case


def bandit_batch() -> EstimatorContext:
    mdp, policy = bandit_case()
    trajectories = (
        Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),
        Trajectory((Step(0, "s0", "B", 0.0, "done", True),)),
        Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),
    )
    return EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)


def two_step_batch() -> EstimatorContext:
    mdp, policy = two_step_case()
    trajectories = (
        Trajectory(
            (
                Step(0, "s0", "LEFT", 0.0, "s_left", False),
                Step(1, "s_left", "FINISH", 2.0, "done", True),
            )
        ),
        Trajectory(
            (
                Step(0, "s0", "RIGHT", 1.0, "s_right", False),
                Step(1, "s_right", "FINISH", 0.0, "done", True),
            )
        ),
    )
    return EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)


def test_oracle_advantage_matches_exact_advantages():
    context = bandit_batch()
    credits = OracleAdvantage().estimate(context)
    assert credits[0] == pytest.approx((0.5,))
    assert credits[1] == pytest.approx((-0.5,))
    assert credits[2] == pytest.approx((0.5,))


def test_outcome_broadcast_assigns_total_return_to_every_step():
    context = two_step_batch()
    credits = OutcomeBroadcast().estimate(context)
    assert credits[0] == pytest.approx((2.0, 2.0))
    assert credits[1] == pytest.approx((1.0, 1.0))


def test_batch_centered_uses_leave_one_out_mean():
    context = bandit_batch()  # returns are 1, 0, 1
    credits = BatchCenteredBroadcast().estimate(context)
    assert credits[0] == pytest.approx((1.0 - 0.5,))
    assert credits[1] == pytest.approx((0.0 - 1.0,))
    assert credits[2] == pytest.approx((1.0 - 0.5,))


def test_batch_centered_single_trajectory():
    mdp, policy = bandit_case()
    context = EstimatorContext(
        mdp=mdp,
        policy=policy,
        trajectories=(Trajectory((Step(0, "s0", "A", 1.0, "done", True),)),),
    )
    # Decision (documented in batch_centered.py): the leave-one-out mean is
    # undefined for a group of one, so this is a caller error, not a value.
    with pytest.raises(ValueError):
        BatchCenteredBroadcast().estimate(context)


@pytest.mark.parametrize(
    "estimator",
    [OracleAdvantage(), OutcomeBroadcast(), BatchCenteredBroadcast()],
    ids=lambda e: e.name,
)
def test_output_shape_and_finiteness(estimator):
    context = two_step_batch()
    credits = estimator.estimate(context)

    assert isinstance(estimator.name, str) and estimator.name
    assert len(credits) == len(context.trajectories)
    for row, trajectory in zip(credits, context.trajectories, strict=True):
        assert len(row) == len(trajectory.steps)
        assert all(math.isfinite(value) for value in row)
