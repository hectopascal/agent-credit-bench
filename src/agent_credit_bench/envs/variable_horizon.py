"""Variable-horizon diagnostic environment (plan.md §8.3).

Purpose: test estimators when actions decide whether future turns exist.

At every timestep the agent picks STOP (terminate, collect stop_rewards[t])
or CONTINUE (collect continue_reward, move on). The final timestep forces
STOP. There is a single "alive" state; the timestep carries all structure.

With the default stop_rewards (0, 2, 1, 3, 0) and a stop probability of 0.5,
continuing has positive advantage at t=0 and t=2 and negative advantage at
t=1 and t=3 — the required sign mixture (§8.3), pinned in
tests/test_variable_horizon.py.
"""

from dataclasses import dataclass

from agent_credit_bench._validation import validate_probability
from agent_credit_bench.types import Action, State, Transition


@dataclass(frozen=True)
class StopProbabilityPolicy:
    """Stops with fixed probability wherever stopping is optional."""

    stop_probability: float

    def __post_init__(self) -> None:
        validate_probability(self.stop_probability, "stop_probability")

    def action_probabilities(self, timestep, state, actions):
        if list(actions) == ["STOP"]:
            return {"STOP": 1.0}
        return {
            "STOP": self.stop_probability,
            "CONTINUE": 1.0 - self.stop_probability,
        }


@dataclass(frozen=True)
class VariableHorizonEnv:
    stop_rewards: tuple[float, ...] = (0.0, 2.0, 1.0, 3.0, 0.0)
    continue_reward: float = 0.0

    def __post_init__(self) -> None:
        if len(self.stop_rewards) < 2:
            raise ValueError("VariableHorizonEnv needs at least two timesteps")

    @property
    def horizon(self) -> int:
        return len(self.stop_rewards)

    @property
    def initial_state(self) -> State:
        return "alive"

    def states_at(self, timestep: int) -> list[State]:
        return ["alive"]

    def actions(self, timestep: int, state: State) -> list[Action]:
        if timestep == self.horizon - 1:
            return ["STOP"]
        return ["STOP", "CONTINUE"]

    def transitions(
        self, timestep: int, state: State, action: Action
    ) -> list[Transition]:
        if action == "STOP":
            return [Transition("done", self.stop_rewards[timestep], 1.0, True)]
        return [Transition("alive", self.continue_reward, 1.0, False)]
