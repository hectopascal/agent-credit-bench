"""Perfect-score baseline: the exact advantage of each sampled action (plan.md §9.1).

Exists to validate the runner, catch metric bugs, and anchor every figure.
Must score perfectly up to floating-point tolerance.
"""

from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext
from agent_credit_bench.oracle import solve_exact_values


@dataclass(frozen=True)
class OracleAdvantage:
    name: str = "oracle_advantage"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        values = solve_exact_values(context.mdp, context.policy)
        return tuple(
            tuple(
                values.advantages[(step.timestep, step.state, step.action)]
                for step in trajectory.steps
            )
            for trajectory in context.trajectories
        )
