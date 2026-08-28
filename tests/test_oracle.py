"""M1 oracle tests (plan.md §12.1, §13-M1).

Self-contained on purpose: the table-driven MDP and policy live here until a
second test module needs them (scope rule 3 — no abstraction until two
components require it).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from agent_credit_bench import solve_exact_values
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


@dataclass(frozen=True)
class TablePolicy:
    probabilities: Mapping[tuple[int, State], Mapping[Action, float]]

    def action_probabilities(
        self, timestep: int, state: State, actions: Sequence[Action]
    ) -> Mapping[Action, float]:
        return self.probabilities[(timestep, state)]


def bandit_case() -> tuple[TableMDP, TablePolicy]:
    """The plan's first hand calculation (plan.md §13-M1)."""
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "A"): (Transition("done", 1.0, 1.0, True),),
            (0, "s0", "B"): (Transition("done", 0.0, 1.0, True),),
        },
    )
    policy = TablePolicy({(0, "s0"): {"A": 0.5, "B": 0.5}})
    return mdp, policy


def two_step_case() -> tuple[TableMDP, TablePolicy]:
    """Two-step deterministic MDP: reward now (RIGHT) vs bigger reward later (LEFT)."""
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
    policy = TablePolicy(
        {
            (0, "s0"): {"LEFT": 0.25, "RIGHT": 0.75},
            (1, "s_left"): {"FINISH": 1.0},
            (1, "s_right"): {"FINISH": 1.0},
        }
    )
    return mdp, policy


def stochastic_case() -> tuple[TableMDP, TablePolicy]:
    """One-step choice between a gamble and a sure thing."""
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
    policy = TablePolicy({(0, "s0"): {"GAMBLE": 0.5, "SAFE": 0.5}})
    return mdp, policy


def test_bandit_hand_calculation():
    mdp, policy = bandit_case()
    values = solve_exact_values(mdp, policy)

    # Expected values worked out in plan.md §13-M1 / §17.
    assert values.action_values[(0, "s0", "A")] == pytest.approx(1.0)
    assert values.action_values[(0, "s0", "B")] == pytest.approx(0.0)
    assert values.state_values[(0, "s0")] == pytest.approx(0.5)
    assert values.advantages[(0, "s0", "A")] == pytest.approx(0.5)
    assert values.advantages[(0, "s0", "B")] == pytest.approx(-0.5)


def test_two_step_hand_calculation():
    mdp, policy = two_step_case()
    values = solve_exact_values(mdp, policy)

    # Work these out on paper from the transition table above, backwards from
    # t=1 (plan.md §3). The point is that you can predict the oracle, not that
    # the oracle predicts you. Replace each ... with your number.
    assert values.state_values[(1, "s_left")] == pytest.approx(2.0)
    assert values.state_values[(1, "s_right")] == pytest.approx(0.0)
    assert values.action_values[(0, "s0", "LEFT")] == pytest.approx(2.0)
    assert values.action_values[(0, "s0", "RIGHT")] == pytest.approx(1.0)
    assert values.state_values[(0, "s0")] == pytest.approx(1.25)
    assert values.advantages[(0, "s0", "LEFT")] == pytest.approx(0.75)
    assert values.advantages[(0, "s0", "RIGHT")] == pytest.approx(-0.25)


def test_stochastic_hand_calculation():
    mdp, policy = stochastic_case()
    values = solve_exact_values(mdp, policy)

    assert values.action_values[(0, "s0", "GAMBLE")] == pytest.approx(3.0)
    assert values.action_values[(0, "s0", "SAFE")] == pytest.approx(2.0)
    assert values.state_values[(0, "s0")] == pytest.approx(2.5)
    assert values.advantages[(0, "s0", "GAMBLE")] == pytest.approx(0.5)
    assert values.advantages[(0, "s0", "SAFE")] == pytest.approx(-0.5)


def test_invalid_transition_probabilities_raise():
    mdp = TableMDP(
        horizon=1,
        initial_state="s0",
        table={
            (0, "s0", "A"): (
                Transition("x", 1.0, 0.6, True),
                Transition("y", 0.0, 0.6, True),
            ),
        },
    )
    policy = TablePolicy({(0, "s0"): {"A": 1.0}})
    with pytest.raises(ValueError):
        solve_exact_values(mdp, policy)


@pytest.mark.parametrize(
    "case",
    [bandit_case, two_step_case, stochastic_case],
    ids=["bandit", "two_step", "stochastic"],
)
def test_policy_weighted_advantage_is_zero(case):
    """sum_a pi(a|s) A(s,a) == 0 at every reachable state (plan.md §12.1)."""
    mdp, policy = case()
    values = solve_exact_values(mdp, policy)

    for t in range(mdp.horizon):
        for s in mdp.states_at(t):
            actions = mdp.actions(t, s)
            probs = policy.action_probabilities(t, s, actions)
            weighted = sum(probs[a] * values.advantages[(t, s, a)] for a in actions)
            assert weighted == pytest.approx(0.0, abs=1e-9)
