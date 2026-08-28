"""Variable-horizon sweep — the flagship diagnostic (plan.md §11.3).

The question: turn-conditioned LOO baselines condition on trajectory survival
at t. Does that introduce gradient direction bias, or only change credit
values and variance? Trajectory-centered credit is the comparison; exact
advantage is the anchor. Either answer is informative — measure, don't
presuppose.

Usage:
    python experiments/variable_horizon_sweep.py --batch-size 500 --num-seeds 10
"""

import argparse
import csv
from pathlib import Path

from agent_credit_bench.benchmark import run_benchmark
from agent_credit_bench.envs.variable_horizon import (
    StopProbabilityPolicy,
    VariableHorizonEnv,
)
from agent_credit_bench.estimators import BatchCenteredBroadcast, OracleAdvantage
from agent_credit_bench.estimators.turn_loo import TurnLOO

STOP_PROBABILITIES = (0.2, 0.35, 0.5, 0.65, 0.8)
ESTIMATORS = (OracleAdvantage(), BatchCenteredBroadcast(), TurnLOO())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'matplotlib missing — run: pip install -e ".[experiments]"'
        ) from None

    env = VariableHorizonEnv()
    rows = []
    results = {}  # (estimator, p) -> BenchmarkResult
    for p in STOP_PROBABILITIES:
        policy = StopProbabilityPolicy(p)
        for estimator in ESTIMATORS:
            result = run_benchmark(
                mdp=env,
                policy=policy,
                estimator=estimator,
                batch_size=args.batch_size,
                seeds=range(args.num_seeds),
            )
            results[(estimator.name, p)] = result
            for row in result.rows():
                rows.append({"stop_probability": p, **row})
        print(f"stop_probability={p}: done")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "variable_horizon_sweep.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    # Figure 1: gradient direction bias and variance vs stop probability.
    fig, (ax_bias, ax_var) = plt.subplots(1, 2, figsize=(11, 4.5))
    for estimator in ESTIMATORS:
        bias = [
            results[(estimator.name, p)].gradient_direction_bias
            for p in STOP_PROBABILITIES
        ]
        variance = [
            results[(estimator.name, p)].gradient_variance
            for p in STOP_PROBABILITIES
        ]
        ax_bias.plot(STOP_PROBABILITIES, bias, marker="o", label=estimator.name)
        ax_var.plot(STOP_PROBABILITIES, variance, marker="o", label=estimator.name)
    ax_bias.set_xlabel("policy stop probability")
    ax_bias.set_ylabel("cosine(mean gradient, exact gradient)")
    ax_bias.set_title("Gradient direction bias")
    ax_bias.legend()
    ax_var.set_xlabel("policy stop probability")
    ax_var.set_ylabel("E ||g - mean g||^2")
    ax_var.set_yscale("log")
    ax_var.set_title("Gradient variance")
    fig.suptitle(
        f"Variable horizon (batch={args.batch_size}, seeds={args.num_seeds})"
    )
    fig.tight_layout()
    bias_path = args.out_dir / "variable_horizon_gradient.png"
    fig.savefig(bias_path, dpi=150)
    print(f"wrote {bias_path}")

    # Figure 2: per-turn credit bias at stop probability 0.5.
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for estimator in ESTIMATORS:
        per_turn = results[(estimator.name, 0.5)].per_turn
        timesteps = sorted(per_turn)
        ax.plot(
            timesteps,
            [per_turn[t].bias for t in timesteps],
            marker="o",
            label=estimator.name,
        )
    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean(credit - exact advantage)")
    ax.set_title("Per-turn credit bias at stop probability 0.5")
    ax.legend()
    fig2.tight_layout()
    turn_path = args.out_dir / "variable_horizon_turn_bias.png"
    fig2.savefig(turn_path, dpi=150)
    print(f"wrote {turn_path}")


if __name__ == "__main__":
    main()
