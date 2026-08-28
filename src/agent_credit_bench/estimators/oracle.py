"""Perfect-score baseline: the exact advantage of each sampled action (plan.md §9.1).

Exists to validate the runner, catch metric bugs, and anchor every figure.
Must score perfectly up to floating-point tolerance.
"""

from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class OracleAdvantage:
    name: str = "oracle_advantage"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        # TODO(yan): solve_exact_values(context.mdp, context.policy), then for
        # each trajectory return a tuple of
        # advantages[(step.timestep, step.state, step.action)].
        raise NotImplementedError("M2: OracleAdvantage not implemented yet")
