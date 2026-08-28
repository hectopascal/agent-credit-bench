"""Estimator contract (plan.md §6.5).

Lives here (not in benchmark.py) because three estimators need it in M2 —
scope rule 3 satisfied. M3's "implement estimator context" task is therefore
already done.
"""

from dataclasses import dataclass
from typing import Protocol

from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Trajectory


@dataclass(frozen=True)
class EstimatorContext:
    mdp: FiniteHorizonMDP
    policy: Policy
    trajectories: tuple[Trajectory, ...]


class CreditEstimator(Protocol):
    name: str

    def estimate(
        self,
        context: EstimatorContext,
    ) -> tuple[tuple[float, ...], ...]:
        """Return one credit value for every action in every trajectory.

        output[i][t] is the estimated credit for
        context.trajectories[i].steps[t].
        """
        ...
