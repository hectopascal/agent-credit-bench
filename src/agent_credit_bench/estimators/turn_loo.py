"""Turn-conditioned leave-one-out baseline (plan.md §9.4).

Contract (documented before implementation, as the plan requires):

  - a trajectory is "active at t" when it has a step with timestep t,
    i.e. len(trajectory.steps) > t;
  - credit_i_t = return_i - mean(return_j for j != i active at t);
  - when no *other* trajectory is active at t, credit_i_t = return_i.  The
    leave-one-out baseline is unavailable, so the estimator falls back to
    ordinary raw-return REINFORCE.  Using zero would silently delete this
    timestep's policy-gradient contribution and introduce a survival- and
    batch-size-dependent bias.

Conditioning the baseline on survival at t is exactly what the flagship
variable-horizon experiment (plan.md §11.3) interrogates.
"""

from dataclasses import dataclass

from agent_credit_bench._numerics import centered_values
from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class TurnLOO:
    name: str = "turn_loo"

    def estimate(self, context: EstimatorContext) -> tuple[tuple[float, ...], ...]:
        returns = [t.total_return for t in context.trajectories]
        lengths = [len(t.steps) for t in context.trajectories]
        max_length = max(lengths)

        credits = [[0.0] * length for length in lengths]
        for t in range(max_length):
            active = [i for i, length in enumerate(lengths) if length > t]
            count = len(active)
            if count == 1:
                credits[active[0]][t] = returns[active[0]]
                continue
            centered = centered_values([returns[i] for i in active])
            for i, value in zip(active, centered, strict=True):
                credits[i][t] = value * (count / (count - 1))
        return tuple(tuple(row) for row in credits)
