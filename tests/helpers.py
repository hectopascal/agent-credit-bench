"""Shared test fixtures, promoted once a second module needed them (scope rule 3).

test_oracle.py predates this file and keeps its local copies; migrating it to
import from here is optional cleanup.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent_credit_bench.policy import TabularPolicy
from agent_credit_bench.types import Action, State, Transition


@dataclass(frozen=True)
class TableMDP:
    """A FiniteHorizonMDP defined by a literal transition table.

    Keys are (timestep, state, action); values are the transition tuple.
    """

    horizon: int
    initial_state: State
    table: Mapping[tuple[int, State, Action], tuple[Transition, ...]]

    def states_at(self, timestep: int) -> Sequence[State]:
        seen: list[State] = []
        for t, state, _action in self.table:
            if t == timestep and state not in seen:
                seen.append(state)
        return seen

    def actions(self, timestep: int, state: State) -> Sequence[Action]:
        return [a for t, s, a in self.table if t == timestep and s == state]

    def transitions(
        self, timestep: int, state: State, action: Action
    ) -> Sequence[Transition]:
        return self.table[(timestep, state, action)]


def bandit_case() -> tuple[TableMDP, TabularPolicy]:
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "A"): (Transition("done", 1.0, 1.0, True),),
            (0, "s0", "B"): (Transition("done", 0.0, 1.0, True),),
        },
    )
    policy = TabularPolicy({(0, "s0"): {"A": 0.5, "B": 0.5}})
    return mdp, policy


def two_step_case() -> tuple[TableMDP, TabularPolicy]:
    mdp = TableMDP(
        horizon=2,
        initial_state="s0",
        table={
            (0, "s0", "LEFT"): (Transition("s_left", 0.0, 1.0, False),),
            (0, "s0", "RIGHT"): (Transition("s_right", 1.0, 1.0, False),),
            (1, "s_left", "FINISH"): (Transition("done", 2.0, 1.0, True),),
            (1, "s_right", "FINISH"): (Transition("done", 0.0, 1.0, True),),
        },
    )
    policy = TabularPolicy(
        {
            (0, "s0"): {"LEFT": 0.25, "RIGHT": 0.75},
            (1, "s_left"): {"FINISH": 1.0},
            (1, "s_right"): {"FINISH": 1.0},
        }
    )
    return mdp, policy


def stochastic_case() -> tuple[TableMDP, TabularPolicy]:
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "GAMBLE"): (
                Transition("win", 10.0, 0.3, True),
                Transition("lose", 0.0, 0.7, True),
            ),
            (0, "s0", "SAFE"): (Transition("done", 2.0, 1.0, True),),
        },
    )
    policy = TabularPolicy({(0, "s0"): {"GAMBLE": 0.5, "SAFE": 0.5}})
    return mdp, policy
