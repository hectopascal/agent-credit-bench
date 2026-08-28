"""Finite-horizon MDP interface (plan.md §6.1)."""

from collections.abc import Sequence
from typing import Protocol

from agent_credit_bench.types import Action, State, Transition


class FiniteHorizonMDP(Protocol):
    horizon: int
    initial_state: State

    def states_at(self, timestep: int) -> Sequence[State]:
        """Return every state reachable at this timestep."""
        ...

    def actions(self, timestep: int, state: State) -> Sequence[Action]:
        """Return available actions."""
        ...

    def transitions(
        self,
        timestep: int,
        state: State,
        action: Action,
    ) -> Sequence[Transition]:
        """Return the transition distribution."""
        ...
