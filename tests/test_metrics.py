"""M3 metric tests (plan.md §10, §12.5, §13-M3)."""

import math

import pytest

from agent_credit_bench.estimators import OracleAdvantage
from agent_credit_bench.estimators.base import EstimatorContext
from agent_credit_bench.gradients import (
    batch_gradient,
    cosine_similarity,
    exact_policy_gradient,
    gradient_variance,
    mean_gradient,
    state_visitation,
)
from agent_credit_bench.metrics import (
    centered_rmse,
    leakage,
    per_turn_stats,
    rmse,
    sign_accuracy,
    spearman,
)
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import TabularPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Transition
from helpers import TableMDP, bandit_case, stochastic_case, two_step_case


def test_rmse_hand_calculation():
    assert rmse([1.0, 2.0], [0.0, 0.0]) == pytest.approx(math.sqrt(2.5))


def test_rmse_rejects_misaligned_or_empty_input():
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        rmse([], [])


def test_spearman_monotonic_and_constant():
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([3.0, 2.0, 1.0], [10.0, 20.0, 30.0]) == pytest.approx(-1.0)
    # Defined behavior for constant input (plan.md §10.2): 0.0.
    assert spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0


def test_spearman_uses_ranks_for_nonlinear_monotonic_data():
    # Rank correlation is perfect; Pearson correlation is only about 0.785.
    assert spearman(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 100.0],
    ) == pytest.approx(1.0)


def test_spearman_uses_average_ranks_for_ties():
    # Average ranks give -1/6; minimum ranks would give about -0.7763.
    value = spearman(
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
    )
    assert value == pytest.approx(-1 / 6)


def test_sign_accuracy_hand_calculation():
    result = sign_accuracy(
        estimated=[1.0, -1.0, 1.0, 0.5],
        exact=[2.0, -3.0, -1.0, 1e-12],
    )
    # The 1e-12 advantage is excluded; 2 of the 3 scored signs match.
    assert result.num_excluded == 1
    assert result.num_scored == 3
    assert result.accuracy == pytest.approx(2 / 3)


def test_sign_accuracy_all_excluded():
    result = sign_accuracy([1.0], [0.0])
    assert result.accuracy is None
    assert result.num_excluded == 1


def test_leakage_hand_calculation():
    result = leakage(estimated=[1.0, -1.0, 2.0], exact=[0.0, 5.0, 0.0])
    assert result.ratio == pytest.approx(3.0 / 4.0, abs=1e-6)
    assert result.mean_abs_zero_credit == pytest.approx(1.5)
    assert result.num_zero_advantage == 2


def test_centered_rmse_is_shift_invariant():
    """Adding b(t, s) to exact advantage must give centered error zero (§12.5)."""
    exact = [0.5, -0.5, 0.2, -0.2]
    keys = [(0, "x"), (0, "x"), (1, "y"), (1, "y")]
    shifts = {(0, "x"): 3.0, (1, "y"): -7.0}
    shifted = [a + shifts[k] for a, k in zip(exact, keys, strict=True)]

    plain = rmse(shifted, exact)
    centered = centered_rmse(shifted, exact, keys)
    assert plain > 1.0
    assert centered.value == pytest.approx(0.0, abs=1e-12)
    assert centered.multi_visit_fraction == 1.0


def test_centered_rmse_never_exceeds_rmse_and_reports_coverage():
    estimated = [1.0, 0.0, 2.0, 5.0]
    exact = [0.5, -0.5, 0.2, 0.1]
    keys = [(0, "x"), (0, "x"), (1, "y"), (2, "z")]  # y and z visited once
    centered = centered_rmse(estimated, exact, keys)
    assert centered.value <= rmse(estimated, exact) + 1e-12
    assert centered.multi_visit_fraction == pytest.approx(0.5)


def test_per_turn_stats_hand_calculation():
    stats = per_turn_stats(
        timesteps=[0, 0, 1],
        estimated=[1.0, 3.0, 5.0],
        exact=[0.0, 0.0, 5.0],
    )
    assert stats[0].bias == pytest.approx(2.0)
    assert stats[0].variance == pytest.approx(1.0)  # values 1, 3 around mean 2
    assert stats[0].count == 2
    assert stats[1].bias == pytest.approx(0.0)
    assert stats[1].count == 1


def test_gradient_variance_uses_population_denominator():
    key = (0, "s0", "A")
    gradients = [{key: 0.0}, {key: 2.0}]
    # The mean is 1 and E[(g - mean)^2] = (1 + 1) / 2 = 1.
    assert gradient_variance(gradients) == pytest.approx(1.0)


def test_visitation_two_step_case():
    mdp, policy = two_step_case()
    visitation = state_visitation(mdp, policy)
    assert visitation[(0, "s0")] == pytest.approx(1.0)
    assert visitation[(1, "s_left")] == pytest.approx(0.25)
    assert visitation[(1, "s_right")] == pytest.approx(0.75)


def test_visitation_drops_terminated_mass():
    """d_t sums to the alive probability mass at each timestep (§12.5)."""
    mdp = TableMDP(
        horizon=2,
        initial_state="s0",
        table={
            (0, "s0", "STOP"): (Transition("done", 0.0, 1.0, True),),
            (0, "s0", "GO"): (Transition("s1", 0.0, 1.0, False),),
            (1, "s1", "END"): (Transition("done", 1.0, 1.0, True),),
        },
    )
    policy = TabularPolicy(
        {
            (0, "s0"): {"STOP": 0.5, "GO": 0.5},
            (1, "s1"): {"END": 1.0},
        }
    )
    visitation = state_visitation(mdp, policy)
    assert visitation[(0, "s0")] == pytest.approx(1.0)
    assert visitation[(1, "s1")] == pytest.approx(0.5)


def test_exact_gradient_bandit_hand_calculation():
    mdp, policy = bandit_case()
    values = solve_exact_values(mdp, policy)
    g_star = exact_policy_gradient(mdp, policy, values)
    # g* = d * pi * A = 1.0 * 0.5 * (+-0.5)
    assert g_star[(0, "s0", "A")] == pytest.approx(0.25)
    assert g_star[(0, "s0", "B")] == pytest.approx(-0.25)
    assert cosine_similarity(g_star, g_star) == pytest.approx(1.0)


def _oracle_credit_gradients(mdp, policy, seeds, batch_size=2000):
    estimator = OracleAdvantage()
    gradients = []
    for seed in seeds:
        trajectories = sample_trajectories(mdp, policy, batch_size, seed)
        context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
        credits = estimator.estimate(context)
        gradients.append(batch_gradient(mdp, policy, trajectories, credits))
    return gradients


def test_oracle_credit_reproduces_exact_gradient():
    """Mean batch gradient of oracle credit converges to g* (§12.5)."""
    mdp, policy = stochastic_case()
    values = solve_exact_values(mdp, policy)
    g_star = exact_policy_gradient(mdp, policy, values)
    gradients = _oracle_credit_gradients(mdp, policy, seeds=range(5))
    assert cosine_similarity(mean_gradient(gradients), g_star) > 0.99


def test_state_shift_leaves_expected_gradient_unchanged():
    """Credit shifted by b(t, s) has the same expected gradient (§12.5)."""
    mdp, policy = stochastic_case()
    values = solve_exact_values(mdp, policy)
    g_star = exact_policy_gradient(mdp, policy, values)

    estimator = OracleAdvantage()
    gradients = []
    for seed in range(5):
        trajectories = sample_trajectories(mdp, policy, 2000, seed)
        context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
        credits = estimator.estimate(context)
        shifted = tuple(
            tuple(value + 3.0 for value in row) for row in credits  # b(0, s0) = 3
        )
        gradients.append(batch_gradient(mdp, policy, trajectories, shifted))
    assert cosine_similarity(mean_gradient(gradients), g_star) > 0.99
