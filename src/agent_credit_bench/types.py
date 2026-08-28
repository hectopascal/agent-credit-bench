"""Core value types (plan.md §6.1, §6.3, §6.4)."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeAlias

State: TypeAlias = Hashable
Action: TypeAlias = Hashable


@dataclass(frozen=True)
class Transition:
    next_state: State
    reward: float
    probability: float
    terminated: bool


@dataclass(frozen=True)
class Step:
    timestep: int
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[Step, ...]

    @property
    def total_return(self) -> float:
        return sum(step.reward for step in self.steps)


@dataclass(frozen=True)
class ExactValues:
    state_values: dict[tuple[int, "State"], float]
    action_values: dict[tuple[int, "State", "Action"], float]
    advantages: dict[tuple[int, "State", "Action"], float]
