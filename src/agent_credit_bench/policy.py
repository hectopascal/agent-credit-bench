"""Policy interface and simple concrete policies (plan.md §6.2, §13-M2).

Policies are fixed during evaluation. The benchmark does not train them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from agent_credit_bench.types import Action, State


class Policy(Protocol):
    def action_probabilities(
        self,
        timestep: int,
        state: State,
        actions: Sequence[Action],
    ) -> Mapping[Action, float]:
        """Return a valid probability distribution over actions."""
        ...


@dataclass(frozen=True)
class TabularPolicy:
    """A policy defined by a literal (timestep, state) -> {action: prob} table."""

    probabilities: Mapping[tuple[int, State], Mapping[Action, float]]

    def action_probabilities(
        self,
        timestep: int,
        state: State,
        actions: Sequence[Action],
    ) -> Mapping[Action, float]:
        return self.probabilities[(timestep, state)]


@dataclass(frozen=True)
class UniformPolicy:
    """Uniform over whatever actions are available."""

    def action_probabilities(
        self,
        timestep: int,
        state: State,
        actions: Sequence[Action],
    ) -> Mapping[Action, float]:
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}
