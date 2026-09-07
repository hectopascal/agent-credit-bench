"""Independent rational-oracle and exhaustive baseline checks.

Dyadic probabilities and seeded cases keep failures reproducible.
"""

import itertools
import random
from fractions import Fraction as F

from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    OutcomeBroadcast,
    TurnLOO,
)
from agent_credit_bench.gradients import batch_gradient, exact_policy_gradient
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import TabularPolicy
from agent_credit_bench.types import Step, Trajectory, Transition
from helpers import TableMDP


def rational_oracle(mdp, policy):
    v, q, a = {}, {}, {}
    for t in reversed(range(mdp.horizon)):
        for s in mdp.states_at(t):
            actions = mdp.actions(t, s)
            probs = policy.action_probabilities(t, s, actions)
            for action in actions:
                q[t, s, action] = sum(
                    F(tr.probability)
                    * (
                        F(tr.reward)
                        + (
                            F(0)
                            if tr.terminated or t + 1 == mdp.horizon
                            else v[t + 1, tr.next_state]
                        )
                    )
                    for tr in mdp.transitions(t, s, action)
                    if tr.probability
                )
            v[t, s] = sum(F(probs[action]) * q[t, s, action] for action in actions)
            for action in actions:
                a[t, s, action] = q[t, s, action] - v[t, s]
    visits = {(0, mdp.initial_state): F(1)}
    for t in range(mdp.horizon - 1):
        for s in mdp.states_at(t):
            for action, p in policy.probabilities[t, s].items():
                for tr in mdp.transitions(t, s, action):
                    if tr.terminated or not p or not tr.probability:
                        continue
                    key = (t + 1, tr.next_state)
                    visits[key] = visits.get(key, F(0)) + visits.get((t, s), F(0)) * F(
                        p
                    ) * F(tr.probability)
    g = {
        key: visits.get(key[:2], F(0)) * F(policy.probabilities[key[:2]][key[2]]) * adv
        for key, adv in a.items()
    }
    return v, q, a, g


def paths(mdp, policy, t=0, state="root", mass=1.0, steps=()):
    for action, p in policy.probabilities[t, state].items():
        for tr in mdp.transitions(t, state, action):
            weight = mass * p * tr.probability
            if not weight:
                continue
            step = Step(t, state, action, tr.reward, tr.next_state, tr.terminated)
            history = steps + (step,)
            if tr.terminated or t + 1 == mdp.horizon:
                yield weight, Trajectory(history)
            else:
                yield from paths(mdp, policy, t + 1, tr.next_state, weight, history)


def test_random_oracle_and_exhaustive_estimator_gradients():
    rng = random.Random(987625)
    worst = 0.0
    for case in range(3000):
        horizon = rng.randint(1, 4)
        table, probs = {}, {}
        for t in range(horizon):
            for state in ["root"] if t == 0 else ["left", "right"]:
                first = rng.randint(0, 8)
                second = rng.randint(0, 8 - first)
                probs[t, state] = dict(
                    zip(
                        "ABC", [first / 8, second / 8, (8 - first - second) / 8],
                        strict=True,
                    )
                )
                for action in "ABC":
                    p = rng.randint(0, 8) / 8
                    table[t, state, action] = tuple(
                        Transition(
                            ns,
                            rng.randint(-40, 40) / 7,
                            w,
                            t + 1 == horizon or rng.random() < 0.25,
                        )
                        for ns, w in [("left", p), ("right", 1 - p)]
                    )
        mdp = TableMDP(horizon, "root", table)
        policy = TabularPolicy(probs)
        values = solve_exact_values(mdp, policy)
        reference = rational_oracle(mdp, policy)
        actual = (
            values.state_values,
            values.action_values,
            values.advantages,
            exact_policy_gradient(mdp, policy, values),
        )
        for expected, got in zip(reference, actual, strict=True):
            for key in expected.keys() | got.keys():
                error = abs(float(expected.get(key, 0)) - got.get(key, 0))
                worst = max(worst, error)
                assert error < 2e-13, (case, key, error)
        reversed_mdp = TableMDP(horizon, "root", dict(reversed(list(table.items()))))
        reordered = solve_exact_values(reversed_mdp, policy)
        assert all(
            abs(reordered.advantages[k] - v) < 2e-13
            for k, v in values.advantages.items()
        )

    worst = 0.0
    for case in range(40):
        table, probs = {}, {}
        for t, state in [(0, "root"), (1, "next")]:
            p = rng.randint(0, 8) / 8
            probs[t, state] = {"A": p, "B": 1 - p}
            for action in "AB":
                q = rng.randint(0, 8) / 8
                table[t, state, action] = tuple(
                    Transition("next", rng.randint(-3, 3) / 4, w, t == 1 or i == 0)
                    for i, w in enumerate([q, 1 - q])
                )
        mdp, policy = TableMDP(2, "root", table), TabularPolicy(probs)
        exact = rational_oracle(mdp, policy)[-1]
        outcomes = list(paths(mdp, policy))
        for size in [1, 2, 3]:
            estimators = [OutcomeBroadcast(), TurnLOO()]
            if size > 1:
                estimators.append(BatchCenteredBroadcast())
            for estimator in estimators:
                expected = {}
                for batch in itertools.product(outcomes, repeat=size):
                    mass = 1.0
                    for p, _ in batch:
                        mass *= p
                    trajectories = tuple(tr for _, tr in batch)
                    context = EstimatorContext(mdp, policy, trajectories)
                    gradient = batch_gradient(
                        mdp, policy, trajectories, estimator.estimate(context)
                    )
                    for key, value in gradient.items():
                        expected[key] = expected.get(key, 0.0) + mass * value
                for key in exact.keys() | expected.keys():
                    error = abs(float(exact.get(key, 0)) - expected.get(key, 0.0))
                    worst = max(worst, error)
                    assert error < 2e-13, (case, size, estimator.name, key, error)
