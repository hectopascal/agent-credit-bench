"""Policy-gradient alignment — gradient-family metrics (plan.md §10.7).

Tabular softmax parameterization with one logit per (t, s, a):

    grad_{theta[t,s,a']} log pi(a | s, t) = 1[a' == a] - pi(a' | s, t)

Everything is closed-form; no autodiff dependency (scope rule 12). Gradient
vectors are sparse dicts keyed by (t, s, a); absent keys are zero.
"""

import math
from collections.abc import Sequence

from agent_credit_bench._validation import (
    validate_positive_integer,
    validated_policy_probabilities,
    validated_transitions,
)
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import Action, ExactValues, State, Trajectory

GradientVector = dict[tuple[int, State, Action], float]


def state_visitation(
    mdp: FiniteHorizonMDP, policy: Policy
) -> dict[tuple[int, State], float]:
    """Exact on-policy state-visitation probability by forward induction.

    Terminated transitions drop their probability mass, so the values at
    timestep t sum to the probability of still being alive at t.
    """
    validate_positive_integer(mdp.horizon, "mdp.horizon")
    visitation: dict[tuple[int, State], float] = {(0, mdp.initial_state): 1.0}
    for t in range(mdp.horizon - 1):
        for s in mdp.states_at(t):
            mass = visitation.get((t, s), 0.0)
            if mass == 0.0:
                continue
            actions = mdp.actions(t, s)
            probs = validated_policy_probabilities(policy, t, s, actions)
            for a in actions:
                transitions = validated_transitions(mdp.transitions(t, s, a), t, s, a)
                for tr in transitions:
                    if tr.terminated or tr.probability == 0.0 or probs[a] == 0.0:
                        continue
                    key = (t + 1, tr.next_state)
                    visitation[key] = (
                        visitation.get(key, 0.0)
                        + mass * probs[a] * tr.probability
                    )
    return visitation


def exact_policy_gradient(
    mdp: FiniteHorizonMDP,
    policy: Policy,
    values: ExactValues,
    visitation: dict[tuple[int, State], float] | None = None,
) -> GradientVector:
    """The exact policy gradient g* under the tabular softmax parameterization.

    g*[t,s,a'] = d(t,s) * sum_a pi(a|s) A(t,s,a) * (1[a'=a] - pi(a'|s))
               = d(t,s) * pi(a'|s) * (A(t,s,a') - sum_a pi(a|s) A(t,s,a))

    The cross term is mathematically zero for exact advantages, but retaining
    it removes residual baseline shifts from floating-point rounding. Center
    around a reference advantage to avoid cancellation under large shifts.
    """
    if visitation is None:
        visitation = state_visitation(mdp, policy)
    gradient: GradientVector = {}
    for (t, s), mass in visitation.items():
        actions = mdp.actions(t, s)
        probs = validated_policy_probabilities(policy, t, s, actions)
        mean = math.fsum(probs[a] * values.advantages[(t, s, a)] for a in actions)
        reference = min(
            (values.advantages[(t, s, a)] for a in actions if probs[a] > 0.0),
            key=lambda advantage: (abs(advantage - mean), advantage),
        )
        differences = {a: values.advantages[(t, s, a)] - reference for a in actions}
        mean_difference = math.fsum(probs[a] * differences[a] for a in actions)
        for a in actions:
            gradient[(t, s, a)] = mass * probs[a] * (differences[a] - mean_difference)
    return gradient


def batch_gradient(
    mdp: FiniteHorizonMDP,
    policy: Policy,
    trajectories: Sequence[Trajectory],
    credits: Sequence[Sequence[float]],
) -> GradientVector:
    """The policy gradient a batch of estimated credit induces.

    g_hat = (1/B) sum_i sum_t credit[i][t] * grad log pi(a_{i,t} | s_{i,t}, t)
    """
    gradient: GradientVector = {}
    batch_size = len(trajectories)
    for trajectory, row in zip(trajectories, credits, strict=True):
        for step, credit in zip(trajectory.steps, row, strict=True):
            t, s = step.timestep, step.state
            actions = mdp.actions(t, s)
            probs = validated_policy_probabilities(policy, t, s, actions)
            for a in actions:
                indicator = 1.0 if a == step.action else 0.0
                key = (t, s, a)
                gradient[key] = (
                    gradient.get(key, 0.0)
                    + credit * (indicator - probs[a]) / batch_size
                )
    return gradient


def norm(gradient: GradientVector) -> float:
    return math.hypot(*gradient.values())


def cosine_similarity(
    g1: GradientVector, g2: GradientVector
) -> float | None:
    """Cosine of two gradient vectors, or ``None`` when either is zero.

    A zero vector has no direction. Returning a numeric sentinel such as 0.0
    would incorrectly describe it as orthogonal to the other vector.
    Nonfinite norms or cosine values raise ValueError rather than being clamped
    into an apparently valid alignment score.
    """
    n1, n2 = norm(g1), norm(g2)
    if not math.isfinite(n1) or not math.isfinite(n2):
        raise ValueError("cosine similarity requires finite gradient norms")
    if n1 == 0.0 or n2 == 0.0:
        return None
    keys = set(g1) | set(g2)
    cosine = math.fsum((g1.get(k, 0.0) / n1) * (g2.get(k, 0.0) / n2) for k in keys)
    if not math.isfinite(cosine):
        raise ValueError("cosine similarity must be finite")
    # Roundoff can put an exactly aligned result a few ulps outside the
    # mathematical range (for example 1.0000000000000002).
    return max(-1.0, min(1.0, cosine))


def mean_gradient(gradients: Sequence[GradientVector]) -> GradientVector:
    keys = set().union(*gradients) if gradients else set()
    n = len(gradients)
    return {k: math.fsum(g.get(k, 0.0) for g in gradients) / n for k in keys}


def gradient_variance(gradients: Sequence[GradientVector]) -> float:
    """Mean squared deviation around the mean gradient: E ||g - mean g||^2."""
    center = mean_gradient(gradients)
    keys = set(center)
    total = math.fsum(
        math.fsum((g.get(k, 0.0) - center[k]) ** 2 for k in keys)
        for g in gradients
    )
    return total / len(gradients)
