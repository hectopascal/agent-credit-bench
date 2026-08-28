"""Trajectory sampling (plan.md §13-M2).

Standard-library random only — the suite keeps zero runtime dependencies.
"""

import random

from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Step, Trajectory


def sample_trajectories(
    mdp: FiniteHorizonMDP,
    policy: Policy,
    batch_size: int,
    seed: int,
) -> tuple[Trajectory, ...]:
    """Sample a batch of on-policy trajectories.

    One ``random.Random(seed)`` drives the whole batch, so the same seed
    reproduces the same batch exactly. Each trajectory starts at
    (t=0, mdp.initial_state) and stops after a terminated transition or when
    the horizon is reached.
    """
    rng = random.Random(seed)
    trajectories: list[Trajectory] = []
    for _ in range(batch_size):
        steps: list[Step] = []
        state = mdp.initial_state
        for t in range(mdp.horizon):
            actions = list(mdp.actions(t, state))
            probs = policy.action_probabilities(t, state, actions)
            action = rng.choices(actions, weights=[probs[a] for a in actions])[0]

            transitions = list(mdp.transitions(t, state, action))
            transition = rng.choices(
                transitions, weights=[tr.probability for tr in transitions]
            )[0]

            steps.append(
                Step(
                    timestep=t,
                    state=state,
                    action=action,
                    reward=transition.reward,
                    next_state=transition.next_state,
                    terminated=transition.terminated,
                )
            )
            if transition.terminated:
                break
            state = transition.next_state
        trajectories.append(Trajectory(tuple(steps)))
    return tuple(trajectories)
