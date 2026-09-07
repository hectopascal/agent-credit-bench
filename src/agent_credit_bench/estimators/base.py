"""Estimator contract (plan.md §6.5).

Lives here (not in benchmark.py) because three estimators need it in M2 —
scope rule 3 satisfied. M3's "implement estimator context" task is therefore
already done.
"""

from dataclasses import dataclass
from typing import Protocol

from agent_credit_bench._validation import validate_integer
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Trajectory


@dataclass(frozen=True)
class EstimatorContext:
    mdp: FiniteHorizonMDP
    policy: Policy
    trajectories: tuple[Trajectory, ...]
    # Optional per-batch randomness for stochastic estimators. None preserves
    # their standalone constructor-seed behavior.
    estimator_seed: int | None = None

    def __post_init__(self) -> None:
        if self.estimator_seed is not None:
            validate_integer(self.estimator_seed, "estimator_seed")


class CreditEstimator(Protocol):
    name: str

    def estimate(
        self,
        context: EstimatorContext,
    ) -> tuple[tuple[float, ...], ...]:
        """Return one credit value for every action in every trajectory.

        output[i][t] is the estimated credit for
        context.trajectories[i].steps[t].
        Stochastic estimators should honor context.estimator_seed when supplied.
        """
        ...
