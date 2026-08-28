"""Trajectory sampling (plan.md §13-M2).

Standard-library random only — the suite keeps zero runtime dependencies.
"""

from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Trajectory


def sample_trajectories(
    mdp: FiniteHorizonMDP,
    policy: Policy,
    batch_size: int,
    seed: int,
) -> tuple[Trajectory, ...]:
    """Sample a batch of on-policy trajectories.

    Contract (pinned by tests/test_sampling.py):

      - build one ``random.Random(seed)``; the same seed must reproduce the
        exact same batch;
      - each trajectory starts at (t=0, mdp.initial_state);
      - at each step: draw the action from policy.action_probabilities, then
        draw the transition from its probability weights
        (``rng.choices(population, weights=...)`` handles both draws);
      - record a Step(timestep, state, action, reward, next_state, terminated);
      - stop after a terminated transition, or when t + 1 == mdp.horizon.
    """
    # TODO(yan): implement M2 sampling. Work order:
    #   pytest -k alignment      -> steps chain correctly, horizon respected
    #   pytest -k reproducible   -> seeding works
    #   pytest -k frequencies    -> draws follow the distributions
    #   pytest -k mean_return    -> integration check against your M1 oracle
    raise NotImplementedError("M2: trajectory sampling not implemented yet")
