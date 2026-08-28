"""Reusable benchmark runner (plan.md §13-M3).

One call runs an estimator over seeded batches and returns both metric
families (§10.8): identification (is the credit literally an advantage?) and
gradient validity (does it induce the right training signal?).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from agent_credit_bench.estimators.base import CreditEstimator, EstimatorContext
from agent_credit_bench.gradients import (
    batch_gradient,
    cosine_similarity,
    exact_policy_gradient,
    gradient_variance,
    mean_gradient,
    norm,
)
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.metrics import (
    TurnStats,
    centered_rmse,
    leakage,
    per_turn_stats,
    rmse,
    sign_accuracy,
    spearman,
)
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import Policy
from agent_credit_bench.sampling import sample_trajectories


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    rmse: float
    centered_rmse: float
    multi_visit_fraction: float
    spearman: float
    sign_accuracy: float | None
    sign_num_scored: int
    sign_num_excluded: int
    leakage_ratio: float
    gradient_cosine: float  # this batch's gradient vs g*


@dataclass(frozen=True)
class BenchmarkResult:
    estimator: str
    seed_metrics: tuple[SeedMetrics, ...]
    gradient_direction_bias: float  # cosine(mean g_hat over seeds, g*)
    gradient_magnitude_error: float  # ||mean g_hat - g*|| / ||g*||
    gradient_variance: float  # E ||g_hat - mean g_hat||^2 over seeds
    per_turn: dict[int, TurnStats]  # pooled over all seeds

    def rows(self) -> list[dict]:
        """Stable tabular form (one row per seed) for CSV export."""
        return [
            {
                "estimator": self.estimator,
                "seed": m.seed,
                "rmse": m.rmse,
                "centered_rmse": m.centered_rmse,
                "multi_visit_fraction": m.multi_visit_fraction,
                "spearman": m.spearman,
                "sign_accuracy": m.sign_accuracy,
                "sign_num_scored": m.sign_num_scored,
                "sign_num_excluded": m.sign_num_excluded,
                "leakage_ratio": m.leakage_ratio,
                "gradient_cosine": m.gradient_cosine,
                "gradient_direction_bias": self.gradient_direction_bias,
                "gradient_magnitude_error": self.gradient_magnitude_error,
                "gradient_variance": self.gradient_variance,
            }
            for m in self.seed_metrics
        ]


def run_benchmark(
    mdp: FiniteHorizonMDP,
    policy: Policy,
    estimator: CreditEstimator,
    batch_size: int,
    seeds: Iterable[int],
) -> BenchmarkResult:
    values = solve_exact_values(mdp, policy)
    g_star = exact_policy_gradient(mdp, policy, values)

    seed_metrics: list[SeedMetrics] = []
    batch_gradients = []
    all_timesteps: list[int] = []
    all_estimated: list[float] = []
    all_exact: list[float] = []

    for seed in seeds:
        trajectories = sample_trajectories(mdp, policy, batch_size, seed)
        context = EstimatorContext(
            mdp=mdp, policy=policy, trajectories=trajectories
        )
        credits = estimator.estimate(context)

        estimated: list[float] = []
        exact: list[float] = []
        keys: list[tuple[int, object]] = []
        timesteps: list[int] = []
        for trajectory, row in zip(trajectories, credits, strict=True):
            for step, credit in zip(trajectory.steps, row, strict=True):
                estimated.append(credit)
                exact.append(
                    values.advantages[(step.timestep, step.state, step.action)]
                )
                keys.append((step.timestep, step.state))
                timesteps.append(step.timestep)

        centered = centered_rmse(estimated, exact, keys)
        signs = sign_accuracy(estimated, exact)
        g_hat = batch_gradient(mdp, policy, trajectories, credits)
        batch_gradients.append(g_hat)

        seed_metrics.append(
            SeedMetrics(
                seed=seed,
                rmse=rmse(estimated, exact),
                centered_rmse=centered.value,
                multi_visit_fraction=centered.multi_visit_fraction,
                spearman=spearman(estimated, exact),
                sign_accuracy=signs.accuracy,
                sign_num_scored=signs.num_scored,
                sign_num_excluded=signs.num_excluded,
                leakage_ratio=leakage(estimated, exact).ratio,
                gradient_cosine=cosine_similarity(g_hat, g_star),
            )
        )
        all_timesteps.extend(timesteps)
        all_estimated.extend(estimated)
        all_exact.extend(exact)

    center = mean_gradient(batch_gradients)
    g_star_norm = norm(g_star)
    difference = {
        k: center.get(k, 0.0) - g_star.get(k, 0.0)
        for k in set(center) | set(g_star)
    }
    magnitude_error = (
        norm(difference) / g_star_norm if g_star_norm > 0 else float("inf")
    )

    return BenchmarkResult(
        estimator=estimator.name,
        seed_metrics=tuple(seed_metrics),
        gradient_direction_bias=cosine_similarity(center, g_star),
        gradient_magnitude_error=magnitude_error,
        gradient_variance=gradient_variance(batch_gradients),
        per_turn=per_turn_stats(all_timesteps, all_estimated, all_exact),
    )
