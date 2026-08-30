"""Shared validation for finite-horizon MDP inputs."""

import math
from collections.abc import Mapping, Sequence

from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Action, State, Transition

_PROBABILITY_TOLERANCE = 1e-9


def validate_probability(probability: float, description: str) -> None:
    """Require one finite probability in the closed interval [0, 1]."""
    try:
        finite = math.isfinite(probability)
        in_range = 0.0 <= probability <= 1.0
    except (OverflowError, TypeError) as exc:
        raise ValueError(f"{description} must be a real number in [0, 1]") from exc
    if not finite or not in_range:
        raise ValueError(
            f"{description} must be finite and in [0, 1], got {probability!r}"
        )


def validate_probability_distribution(
    probabilities: Sequence[float], description: str
) -> None:
    """Require a nonempty, normalized sequence of valid probabilities."""
    if not probabilities:
        raise ValueError(f"{description} must not be empty")
    for index, probability in enumerate(probabilities):
        validate_probability(probability, f"{description}[{index}]")
    total = math.fsum(probabilities)
    if not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE
    ):
        raise ValueError(f"{description} sum to {total}, expected 1.0")


def validated_policy_probabilities(
    policy: Policy,
    timestep: int,
    state: State,
    actions: Sequence[Action],
) -> Mapping[Action, float]:
    """Return a policy distribution after validating its available actions."""
    if not actions:
        raise ValueError(
            f"no actions available at (t={timestep}, s={state!r})"
        )
    if len(set(actions)) != len(actions):
        raise ValueError(
            f"duplicate actions at (t={timestep}, s={state!r})"
        )
    probabilities = policy.action_probabilities(timestep, state, actions)
    action_set = set(actions)
    unexpected = [action for action in probabilities if action not in action_set]
    if unexpected:
        raise ValueError(
            f"policy returned probabilities for unavailable actions {unexpected!r} "
            f"at (t={timestep}, s={state!r})"
        )
    try:
        weights = [probabilities[action] for action in actions]
    except KeyError as exc:
        raise ValueError(
            f"policy is missing probability for action {exc.args[0]!r} "
            f"at (t={timestep}, s={state!r})"
        ) from exc
    validate_probability_distribution(
        weights, f"policy probabilities at (t={timestep}, s={state!r})"
    )
    return probabilities


def validated_transitions(
    transitions: Sequence[Transition],
    timestep: int,
    state: State,
    action: Action,
) -> Sequence[Transition]:
    """Return a transition distribution after validating every weight."""
    validate_probability_distribution(
        [transition.probability for transition in transitions],
        f"transition probabilities at (t={timestep}, s={state!r}, a={action!r})",
    )
    return transitions


def validate_integer(value: int, description: str) -> None:
    """Require an integer (booleans are not accepted as integer inputs)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer")


def validate_positive_integer(value: int, description: str) -> None:
    """Require an integer greater than zero (booleans are not counts)."""
    validate_integer(value, description)
    if value <= 0:
        raise ValueError(f"{description} must be greater than zero")
