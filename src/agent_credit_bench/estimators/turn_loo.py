"""Turn-conditioned leave-one-out baseline (plan.md §9.4).

Contract (documented before implementation, as the plan requires):

  - a trajectory is "active at t" when it has a step with timestep t,
    i.e. len(trajectory.steps) > t;
  - credit_i_t = return_i - mean(return_j for j != i active at t);
  - when no *other* trajectory is active at t, credit_i_t = 0.0: with no
    peers there is no baseline information, and emitting the raw return
    instead would smuggle in an uncentered update.

Conditioning the baseline on survival at t is exactly what the flagship
variable-horizon experiment (plan.md §11.3) interrogates.
"""

from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class TurnLOO:
    name: str = "turn_loo"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        returns = [t.total_return for t in context.trajectories]
        lengths = [len(t.steps) for t in context.trajectories]
        max_length = max(lengths)

        # Per-timestep sums over active trajectories, computed once.
        active_sum = [0.0] * max_length
        active_count = [0] * max_length
        for ret, length in zip(returns, lengths, strict=True):
            for t in range(length):
                active_sum[t] += ret
                active_count[t] += 1

        credits = []
        for ret, length in zip(returns, lengths, strict=True):
            row = []
            for t in range(length):
                peers = active_count[t] - 1
                if peers == 0:
                    row.append(0.0)
                else:
                    baseline = (active_sum[t] - ret) / peers
                    row.append(ret - baseline)
            credits.append(tuple(row))
        return tuple(credits)
