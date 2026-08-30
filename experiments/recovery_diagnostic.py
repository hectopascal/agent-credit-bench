"""Recovery credit diagnostic (plan.md §11.2, M4 exit condition).

Conditions on successful BAD -> RECOVER trajectories and reports the average
credit each estimator assigns to the mistake and to the repair. An estimator
that identifies the two actions separates their signs; a broadcast estimator
is positive on both within this selected subset.

Usage:
    python experiments/recovery_diagnostic.py --batch-size 2000 --seed 0
"""

import argparse
import csv
from pathlib import Path
from statistics import fmean, pstdev

from agent_credit_bench.envs.recovery import RecoveryEnv
from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    GiGPOStyle,
    GRPOStyleNormalized,
    OracleAdvantage,
    OutcomeBroadcast,
)
from agent_credit_bench.gradients import (
    batch_gradient,
    cosine_similarity,
    exact_policy_gradient,
    norm,
)
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

ESTIMATORS = (
    OracleAdvantage(),
    OutcomeBroadcast(),
    BatchCenteredBroadcast(),
    GRPOStyleNormalized(),
    GiGPOStyle(),
)


def _stable_value(value: object) -> object:
    """Keep committed numeric artifacts stable across hash seeds/platforms."""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--recover-success", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("results/recovery_credit.png"))
    parser.add_argument(
        "--csv-out", type=Path, default=Path("results/recovery_diagnostic.csv")
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'matplotlib missing — run: pip install -e ".[experiments]"'
        ) from None

    if args.batch_size <= 0 or args.num_seeds <= 0:
        raise SystemExit("--batch-size and --num-seeds must be positive")

    env = RecoveryEnv(recover_success_probability=args.recover_success)
    policy = UniformPolicy()
    values = solve_exact_values(env, policy)
    g_star = exact_policy_gradient(env, policy, values)
    rows: list[dict[str, float | int | str | None]] = []
    g_star_norm = norm(g_star)
    if g_star_norm == 0.0:
        raise RuntimeError("the recovery diagnostic requires a nonzero exact gradient")
    for seed in range(args.seed, args.seed + args.num_seeds):
        trajectories = sample_trajectories(env, policy, args.batch_size, seed)
        context = EstimatorContext(mdp=env, policy=policy, trajectories=trajectories)
        recovered = [
            i
            for i, trajectory in enumerate(trajectories)
            if len(trajectory.steps) == 2
            and trajectory.steps[0].action == "BAD"
            and trajectory.steps[1].action == "RECOVER"
            and trajectory.total_return == 1.0
        ]
        all_bad = [
            i
            for i, trajectory in enumerate(trajectories)
            if trajectory.steps[0].action == "BAD"
        ]
        all_good = [
            i
            for i, trajectory in enumerate(trajectories)
            if trajectory.steps[0].action == "GOOD"
        ]
        if not recovered:
            raise RuntimeError(f"seed {seed} sampled no successful recovery")
        for estimator in ESTIMATORS:
            credits = estimator.estimate(context)
            bad_recovered = [credits[i][0] for i in recovered]
            recover = [credits[i][1] for i in recovered]
            batch_g = batch_gradient(env, policy, trajectories, credits)
            difference = {
                key: batch_g.get(key, 0.0) - g_star.get(key, 0.0)
                for key in set(batch_g) | set(g_star)
            }
            rows.append(
                {
                    "estimator": estimator.name,
                    "batch_size": args.batch_size,
                    "seed": seed,
                    "num_seeds": args.num_seeds,
                    "recover_success_probability": (
                        env.recover_success_probability
                    ),
                    "num_recovered": len(recovered),
                    "mean_credit_bad_on_recovered": fmean(bad_recovered),
                    "mean_credit_recover_on_recovered": fmean(recover),
                    "both_positive_fraction_on_recovered": fmean(
                        b > 0 and r > 0
                        for b, r in zip(bad_recovered, recover, strict=True)
                    ),
                    "mean_credit_all_bad": fmean(credits[i][0] for i in all_bad),
                    "mean_credit_all_good": fmean(credits[i][0] for i in all_good),
                    "batch_gradient_cosine": cosine_similarity(batch_g, g_star),
                    "batch_gradient_relative_error": norm(difference) / g_star_norm,
                }
            )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {key: _stable_value(value) for key, value in row.items()} for row in rows
        )
    print(f"wrote {args.csv_out}")

    aggregate: dict[str, dict[str, float]] = {}
    numeric_fields = (
        "mean_credit_bad_on_recovered",
        "mean_credit_recover_on_recovered",
        "both_positive_fraction_on_recovered",
        "mean_credit_all_bad",
        "mean_credit_all_good",
        "batch_gradient_cosine",
        "batch_gradient_relative_error",
    )
    for estimator in ESTIMATORS:
        selected = [r for r in rows if r["estimator"] == estimator.name]
        aggregate[estimator.name] = {}
        for field in numeric_fields:
            present = [float(row[field]) for row in selected if row[field] is not None]
            aggregate[estimator.name][field] = (
                fmean(present) if present else float("nan")
            )
        aggregate[estimator.name].update(
            {
                f"{field}_std": pstdev(
                    float(row[field]) for row in selected if row[field] is not None
                )
                for field in numeric_fields
                if any(row[field] is not None for row in selected)
            }
        )

    print(
        f"{'estimator':<28}{'recovered BAD':>15}{'RECOVER':>11}"
        f"{'all BAD':>11}{'grad cos':>11}{'rel err':>11}"
    )
    for name, stats in aggregate.items():
        print(
            f"{name:<28}{stats['mean_credit_bad_on_recovered']:>15.4f}"
            f"{stats['mean_credit_recover_on_recovered']:>11.4f}"
            f"{stats['mean_credit_all_bad']:>11.4f}"
            f"{stats['batch_gradient_cosine']:>11.4f}"
            f"{stats['batch_gradient_relative_error']:>11.4f}"
        )

    names = list(aggregate)
    positions = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [p - width / 2 for p in positions],
        [aggregate[n]["mean_credit_bad_on_recovered"] for n in names],
        width,
        label="BAD (exact advantage < 0)",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [aggregate[n]["mean_credit_recover_on_recovered"] for n in names],
        width,
        label="RECOVER (exact advantage > 0)",
    )
    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("mean credit on successful BAD -> RECOVER trajectories")
    ax.set_title(
        f"Recovery diagnostic (batch={args.batch_size}, seeds={args.num_seeds})"
    )
    ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
