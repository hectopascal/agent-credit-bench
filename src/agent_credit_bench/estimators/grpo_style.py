"""GRPO-style normalized group-relative baseline (plan.md §16, item 1).

Faithful reimplementation of the published group-advantage formula
(DeepSeekMath / TRL GRPOTrainer):

    credit_i_t = (return_i - mean(returns)) / (pstdev(returns) + epsilon)

Differences from BatchCenteredBroadcast: the mean includes the trajectory
itself (no leave-one-out), and the result is normalized by the group standard
deviation. Note the normalization makes the output scale-free, so its RMSE
against exact advantage is not meaningful on its own — the centered/gradient
metrics are the fair ones (plan.md §10.8).

A batch whose returns are all equal (std = 0) gets zero credit everywhere,
matching library behavior. Labeled "-style" because it reproduces the
formula, not the library code path; use TrajectoryReturnAdapter to score the
actual library function.
"""

import statistics
from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class GRPOStyleNormalized:
    epsilon: float = 1e-4
    name: str = "grpo_style_normalized"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        returns = [t.total_return for t in context.trajectories]
        mean = statistics.fmean(returns)
        std = statistics.pstdev(returns)
        return tuple(
            tuple((ret - mean) / (std + self.epsilon) for _ in trajectory.steps)
            for ret, trajectory in zip(returns, context.trajectories, strict=True)
        )
