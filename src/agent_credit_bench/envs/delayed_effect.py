"""Delayed-effect diagnostic environment (plan.md §8.1).

Purpose: test whether an estimator leaks delayed terminal reward onto actions
that have no effect on the outcome.

Contract (pinned by tests/test_delayed_effect.py):

  - at t=0 the available actions are exactly "GOOD" and "BAD";
  - the t=0 choice sets the latent success probability
    (good_success_probability or bad_success_probability) and pays reward 0;
  - at t = 1 .. horizon-1 the agent picks among ``num_distractor_actions``
    behaviorally identical actions — identical transitions and rewards — so
    every distractor has exactly zero advantage;
  - all intermediate rewards are 0; the final transition (t == horizon - 1)
    terminates and pays reward 1.0 with the latent success probability,
    else 0.0. Consequence with gamma=1: Q_0(s0, "GOOD") equals
    good_success_probability exactly, under any policy.

The state representation is yours to choose — any hashable values pass the
tests. One natural choice: "start" at t=0, then "good" / "bad" as the latent
state for t >= 1. Requires horizon >= 2.
"""

from dataclasses import dataclass

from agent_credit_bench.types import Action, State, Transition


@dataclass(frozen=True)
class DelayedEffectEnv:
    horizon: int
    good_success_probability: float = 0.8
    bad_success_probability: float = 0.2
    num_distractor_actions: int = 2

    @property
    def initial_state(self) -> State:
        # TODO(yan): return the t=0 state.
        raise NotImplementedError("M2: DelayedEffectEnv not implemented yet")

    def states_at(self, timestep: int) -> list[State]:
        # TODO(yan): every state reachable at this timestep.
        raise NotImplementedError("M2: DelayedEffectEnv not implemented yet")

    def actions(self, timestep: int, state: State) -> list[Action]:
        # TODO(yan): ["GOOD", "BAD"] at t=0; the distractors afterwards.
        raise NotImplementedError("M2: DelayedEffectEnv not implemented yet")

    def transitions(
        self, timestep: int, state: State, action: Action
    ) -> list[Transition]:
        # TODO(yan): see the module docstring for the required structure.
        raise NotImplementedError("M2: DelayedEffectEnv not implemented yet")
