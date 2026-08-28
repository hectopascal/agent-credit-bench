"""Recovery diagnostic environment (plan.md §8.2).

Purpose: test whether an estimator distinguishes an early mistake from the
later action that repairs it.

    t=0: GOOD    -> success, reward 1, done
         BAD     -> "mistake" state, reward 0
    t=1: RECOVER -> success (reward 1) with recover_success_probability, else
                    failure (reward 0); done either way
         GIVE_UP -> failure, reward 0, done

Documented evaluation policy (plan.md §13-M4): uniform. Under it, with
recover_success_probability q:

    Q0(BAD) = q/2        V0 = 1/2 + q/4     A0(BAD)     = q/4 - 1/2  (< 0)
    Q1(RECOVER) = q      V1 = q/2           A1(RECOVER) = q/2        (> 0)

so the oracle sign pattern the diagnostic relies on (BAD negative, RECOVER
positive) holds for any 0 < q <= 1.
"""

from dataclasses import dataclass

from agent_credit_bench.types import Action, State, Transition


@dataclass(frozen=True)
class RecoveryEnv:
    recover_success_probability: float = 1.0
    horizon: int = 2

    def __post_init__(self) -> None:
        if self.horizon != 2:
            raise ValueError("RecoveryEnv is structurally two-step")

    @property
    def initial_state(self) -> State:
        return "s0"

    def states_at(self, timestep: int) -> list[State]:
        return ["s0"] if timestep == 0 else ["mistake"]

    def actions(self, timestep: int, state: State) -> list[Action]:
        return ["GOOD", "BAD"] if timestep == 0 else ["RECOVER", "GIVE_UP"]

    def transitions(
        self, timestep: int, state: State, action: Action
    ) -> list[Transition]:
        if timestep == 0:
            if action == "GOOD":
                return [Transition("success", 1.0, 1.0, True)]
            return [Transition("mistake", 0.0, 1.0, False)]
        if action == "RECOVER":
            q = self.recover_success_probability
            transitions = [Transition("success", 1.0, q, True)]
            if q < 1.0:
                transitions.append(Transition("failure", 0.0, 1.0 - q, True))
            return transitions
        return [Transition("failure", 0.0, 1.0, True)]
