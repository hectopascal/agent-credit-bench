"""Adapter for external trajectory-level advantage code (plan.md §16, item 1).

The adoption bridge: most library implementations (verl, TRL, GRPO/RLOO
reference code) reduce to a function from a batch of trajectory returns to a
batch of trajectory credits. Wrap that function and the suite can score the
*actual* library computation — no reimplementation, no torch dependency in
this package.

Example — scoring TRL-style group whitening in six lines:

    def trl_grpo_whiten(returns):
        import statistics
        mean = statistics.fmean(returns)
        std = statistics.pstdev(returns)
        return [(r - mean) / (std + 1e-4) for r in returns]

    estimator = TrajectoryReturnAdapter(fn=trl_grpo_whiten, name="trl_grpo")

Step-level library code can implement the CreditEstimator protocol directly;
it is a single `estimate` method.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class TrajectoryReturnAdapter:
    """Broadcasts fn(batch returns) -> batch credits onto every step."""

    fn: Callable[[Sequence[float]], Sequence[float]]
    name: str = "trajectory_return_adapter"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        returns = [t.total_return for t in context.trajectories]
        credits = list(self.fn(returns))
        if len(credits) != len(returns):
            raise ValueError(
                f"adapted function returned {len(credits)} credits "
                f"for {len(returns)} trajectories"
            )
        return tuple(
            tuple(float(credit) for _ in trajectory.steps)
            for credit, trajectory in zip(
                credits, context.trajectories, strict=True
            )
        )
