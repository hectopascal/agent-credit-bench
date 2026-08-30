"""Reproduce the TurnLOO no-peer fallback correction.

The legacy implementation emitted zero when a trajectory had no active peer;
the corrected implementation emits the raw return. This focused audit keeps
the original five-step, stop-probability-0.5 protocol and records both rules at
the small group sizes where the difference is visible.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs import StopProbabilityPolicy, VariableHorizonEnv
from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    TurnLOO,
)
from agent_credit_bench.gradients import exact_policy_gradient, norm
from agent_credit_bench.oracle import solve_exact_values


@dataclass(frozen=True)
class LegacyZeroFallbackTurnLOO:
    """The pre-audit rule, retained only to reproduce the corrected claim."""

    name: str = "turn_loo_legacy_zero_fallback"

    def estimate(self, context: EstimatorContext) -> tuple[tuple[float, ...], ...]:
        returns = [trajectory.total_return for trajectory in context.trajectories]
        lengths = [len(trajectory.steps) for trajectory in context.trajectories]
        max_length = max(lengths)
        active_count = [0] * max_length
        active_sum = [0.0] * max_length
        for episode_return, length in zip(returns, lengths, strict=True):
            for timestep in range(length):
                active_count[timestep] += 1
                active_sum[timestep] += episode_return

        rows = []
        for episode_return, length in zip(returns, lengths, strict=True):
            row = []
            for timestep in range(length):
                peer_count = active_count[timestep] - 1
                if peer_count == 0:
                    row.append(0.0)
                else:
                    peer_mean = (active_sum[timestep] - episode_return) / peer_count
                    row.append(episode_return - peer_mean)
            rows.append(tuple(row))
        return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=5000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/turn_loo_fallback_audit.csv"),
    )
    args = parser.parse_args()
    if args.num_seeds <= 0:
        raise SystemExit("--num-seeds must be positive")

    mdp = VariableHorizonEnv()
    policy = StopProbabilityPolicy(0.5)
    exact = solve_exact_values(mdp, policy)
    exact_gradient_norm = norm(exact_policy_gradient(mdp, policy, exact))
    rows = []
    for batch_size in (2, 4, 8):
        for estimator in (
            LegacyZeroFallbackTurnLOO(),
            BatchCenteredBroadcast(),
            TurnLOO(),
        ):
            result = run_benchmark(
                mdp=mdp,
                policy=policy,
                estimator=estimator,
                batch_size=batch_size,
                seeds=range(args.num_seeds),
            )
            relative_error = result.relative_mean_gradient_error
            if relative_error is None or exact_gradient_norm == 0.0:
                raise RuntimeError("fallback audit requires a nonzero exact gradient")
            normalized_variance = result.gradient_variance / exact_gradient_norm**2
            rows.append(
                {
                    "horizon": mdp.horizon,
                    "stop_rewards": "|".join(
                        format(value, ".12g") for value in mdp.stop_rewards
                    ),
                    "continue_reward": mdp.continue_reward,
                    "stop_probability": policy.stop_probability,
                    "batch_size": batch_size,
                    "num_seeds": args.num_seeds,
                    "estimator": estimator.name,
                    "mean_gradient_cosine": result.mean_gradient_cosine,
                    "relative_mean_gradient_error": (
                        relative_error
                    ),
                    "gradient_variance": result.gradient_variance,
                    "normalized_gradient_variance": normalized_variance,
                    "normalized_gradient_mse": (
                        relative_error**2 + normalized_variance
                    ),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: format(value, ".12g") if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in rows
        )
    print(f"wrote {args.out}")
    for row in rows:
        print(
            f"batch={row['batch_size']:>2} {row['estimator']:<31} "
            f"cos={row['mean_gradient_cosine']:.6f} "
            f"relative_error={row['relative_mean_gradient_error']:.6f}"
        )


if __name__ == "__main__":
    main()
