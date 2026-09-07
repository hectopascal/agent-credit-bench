"""Group-relative trajectory baseline with leave-one-out centering (plan.md §9.3).

    credit_i_t = return_i - mean(return_j for j != i)

A simple stand-in for group-relative baselines; not labeled as a complete
GRPO implementation.

Single-trajectory batches raise ValueError: the leave-one-out mean is
undefined, and silently returning something (0, or the raw return) would hide
a caller bug — group-relative credit is meaningless without a group.
"""

from dataclasses import dataclass

from agent_credit_bench._numerics import centered_values
from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class BatchCenteredBroadcast:
    name: str = "batch_centered_broadcast"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        returns = [t.total_return for t in context.trajectories]
        if len(returns) < 2:
            raise ValueError(
                "BatchCenteredBroadcast needs at least two trajectories "
                "for leave-one-out centering"
            )
        centered = centered_values(returns)
        count = len(returns)
        credits = []
        for trajectory, value in zip(context.trajectories, centered, strict=True):
            credit = value * (count / (count - 1))
            credits.append(tuple(credit for _ in trajectory.steps))
        return tuple(credits)
