"""Deliberately naive baseline: broadcast total return to every action (plan.md §9.2).

    credit_t = trajectory.total_return

Exposes reward smearing. Note the output is not on an advantage scale, which
is exactly why the metrics report both value error and gradient validity
(plan.md §10.8).
"""

from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class OutcomeBroadcast:
    name: str = "outcome_broadcast"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(trajectory.total_return for _ in trajectory.steps)
            for trajectory in context.trajectories
        )
