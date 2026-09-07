"""Monte Carlo advantage estimator (plan.md §9.5, §16 item 2).

Estimates Q(s, a) and V(s) at every visited (t, s, a) from sampled
continuations rolled out in the environment model, and returns their
difference. Converges to the exact oracle as num_rollouts grows —
demonstrated in experiments/monte_carlo_convergence.py and pinned loosely in
tests/test_monte_carlo.py.

This is what "approximate ground truth from extra rollouts" costs and buys:
the whole point of the exact oracle is not needing this.

Rollouts are seeded and results are cached per (t, s[, a]) within one
estimate() call. Standalone calls use the constructor seed. When the context
supplies estimator_seed (as run_benchmark does), it is combined with the
constructor seed to resample continuations reproducibly for each batch.
"""

import random
from dataclasses import dataclass

from agent_credit_bench._validation import (
    validate_integer,
    validate_positive_integer,
    validated_policy_probabilities,
    validated_transitions,
)
from agent_credit_bench.estimators.base import EstimatorContext
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Action, State

# None is a legal hashable action; only this private sentinel requests sampling.
_SAMPLE_ACTION = object()


def _rollout(
    mdp: FiniteHorizonMDP,
    policy: Policy,
    rng: random.Random,
    start_timestep: int,
    start_state: State,
    first_action: Action = _SAMPLE_ACTION,
) -> float:
    """Return of one continuation from (t, s), optionally forcing the first action."""
    total = 0.0
    state = start_state
    action = first_action
    for t in range(start_timestep, mdp.horizon):
        actions = list(mdp.actions(t, state))
        if action is _SAMPLE_ACTION:
            probs = validated_policy_probabilities(policy, t, state, actions)
            action = rng.choices(actions, weights=[probs[a] for a in actions])[0]
        transitions = validated_transitions(
            mdp.transitions(t, state, action), t, state, action
        )
        transition = rng.choices(
            transitions, weights=[tr.probability for tr in transitions]
        )[0]
        total += transition.reward
        if transition.terminated:
            break
        state = transition.next_state
        action = _SAMPLE_ACTION
    return total


@dataclass(frozen=True)
class MonteCarloAdvantage:
    num_rollouts: int = 64
    seed: int = 0
    name: str = "monte_carlo_advantage"

    def __post_init__(self) -> None:
        validate_positive_integer(self.num_rollouts, "num_rollouts")
        validate_integer(self.seed, "seed")

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        mdp, policy = context.mdp, context.policy
        rng = random.Random(
            self.seed if context.estimator_seed is None else
            f"agent-credit-bench:mc:v1:{self.seed}:{context.estimator_seed}"
        )
        q_cache: dict[tuple[int, State, Action], float] = {}
        v_cache: dict[tuple[int, State], float] = {}

        def q_value(t: int, s: State, a: Action) -> float:
            key = (t, s, a)
            if key not in q_cache:
                q_cache[key] = sum(
                    _rollout(mdp, policy, rng, t, s, a)
                    for _ in range(self.num_rollouts)
                ) / self.num_rollouts
            return q_cache[key]

        def v_value(t: int, s: State) -> float:
            key = (t, s)
            if key not in v_cache:
                v_cache[key] = sum(
                    _rollout(mdp, policy, rng, t, s)
                    for _ in range(self.num_rollouts)
                ) / self.num_rollouts
            return v_cache[key]

        return tuple(
            tuple(
                q_value(step.timestep, step.state, step.action)
                - v_value(step.timestep, step.state)
                for step in trajectory.steps
            )
            for trajectory in context.trajectories
        )
