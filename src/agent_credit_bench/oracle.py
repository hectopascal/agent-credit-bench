from agent_credit_bench._validation import (
    validate_positive_integer,
    validated_policy_probabilities,
    validated_transitions,
)
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.policy import Policy
from agent_credit_bench.types import ExactValues


def solve_exact_values(mdp: FiniteHorizonMDP, policy: Policy) -> ExactValues:
    """Compute exact V, Q, and A for every reachable (timestep, state, action).

    Backward induction over t = horizon-1 .. 0, with gamma = 1 (plan.md §3):

        Q_t(s, a) = sum over transitions (s', r, p, terminated) of
                        p * (r + future)
                    where future = 0 if terminated or t + 1 == mdp.horizon
                                   else V_{t+1}(s')
        V_t(s)    = sum_a  pi(a | s) * Q_t(s, a)
        A_t(s, a) = Q_t(s, a) - V_t(s)

    Validation (raise ValueError, tolerance ~1e-9):
      - every probability must be finite and in [0, 1];
      - each action's transition probabilities must sum to 1;
      - each state's policy probabilities must sum to 1 over the available
        actions.

    Result keys: state_values[(t, s)], action_values[(t, s, a)],
    advantages[(t, s, a)] for every state in mdp.states_at(t) and every
    available action.
    """
    validate_positive_integer(mdp.horizon, "mdp.horizon")
    V = {}  # state
    Q = {}  # action
    A = {}  # advantage
    for t in range(mdp.horizon - 1, -1, -1):
        for s in mdp.states_at(t):
            actions = mdp.actions(t, s)
            probs = validated_policy_probabilities(policy, t, s, actions)
            V[(t, s)] = 0.0
            for a in actions:
                Q[(t, s, a)] = 0.0
                transitions = mdp.transitions(t, s, a)
                validated_transitions(transitions, t, s, a)
                for transition in transitions:
                    future = (
                        0.0
                        if transition.terminated or t + 1 == mdp.horizon
                        else V[t + 1, transition.next_state]
                    )
                    Q[(t, s, a)] += transition.probability * (
                        transition.reward + future
                    )

                V[(t, s)] += probs[a] * Q[(t, s, a)]

            for a in actions:
                A[(t, s, a)] = Q[(t, s, a)] - V[(t, s)]
    return ExactValues(state_values=V, action_values=Q, advantages=A)
