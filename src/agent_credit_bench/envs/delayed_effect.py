"""Delayed-effect diagnostic environment (plan.md §8.1).

Purpose: test whether an estimator leaks delayed terminal reward onto actions
that have no effect on the outcome.

Structure: at t=0 the agent picks "GOOD" or "BAD", which sets a latent success
probability and pays reward 0. At t = 1 .. horizon-1 it picks among
behaviorally identical distractor actions (identical transitions and rewards,
hence exactly zero advantage). All intermediate rewards are 0; the final
transition terminates and pays 1.0 with the latent success probability, else
0.0. With gamma=1 this makes Q_0(s0, "GOOD") == good_success_probability
exactly, under any policy.

State representation: "start" at t=0, then the latent state "good" / "bad"
for every t >= 1. Requires horizon >= 2.
"""

from dataclasses import dataclass

from agent_credit_bench._validation import (
    validate_positive_integer,
    validate_probability,
)
from agent_credit_bench.types import Action, State, Transition


@dataclass(frozen=True)
class DelayedEffectEnv:
    horizon: int
    good_success_probability: float = 0.8
    bad_success_probability: float = 0.2
    num_distractor_actions: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an integer")
        if self.horizon < 2:
            raise ValueError("DelayedEffectEnv requires horizon >= 2")
        validate_probability(
            self.good_success_probability, "good_success_probability"
        )
        validate_probability(
            self.bad_success_probability, "bad_success_probability"
        )
        validate_positive_integer(
            self.num_distractor_actions, "num_distractor_actions"
        )

    @property
    def initial_state(self) -> State:
        return "start"

    def states_at(self, timestep: int) -> list[State]:
        return ["start"] if timestep == 0 else ["good", "bad"]

    def actions(self, timestep: int, state: State) -> list[Action]:
        if timestep == 0:
            return ["GOOD", "BAD"]
        return [f"DISTRACT_{i}" for i in range(self.num_distractor_actions)]

    def transitions(
        self, timestep: int, state: State, action: Action
    ) -> list[Transition]:
        if timestep == 0:
            latent = "good" if action == "GOOD" else "bad"
            return [Transition(latent, 0.0, 1.0, False)]
        if timestep < self.horizon - 1:
            # Distractors are identical: stay in the latent state, reward 0.
            return [Transition(state, 0.0, 1.0, False)]
        success = (
            self.good_success_probability
            if state == "good"
            else self.bad_success_probability
        )
        return [
            Transition("success", 1.0, success, True),
            Transition("failure", 0.0, 1.0 - success, True),
        ]
