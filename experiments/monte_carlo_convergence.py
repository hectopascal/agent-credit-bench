"""Monte Carlo advantage convergence (plan.md §9.5, §16 item 2).

How many continuation rollouts does approximate ground truth need before it
agrees with the exact oracle? RMSE vs exact advantage as num_rollouts grows,
on the delayed-effect environment.

Usage:
    python experiments/monte_carlo_convergence.py --batch-size 200 --seed 0
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean, pstdev

from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import EstimatorContext, MonteCarloAdvantage
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

ROLLOUT_COUNTS = (1, 4, 16, 64, 256, 1024)
MC_SEED_BASE = 1_000_003
MC_SEED_STRIDE = 7_919


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("results/monte_carlo_convergence.png")
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/monte_carlo_convergence.csv"),
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

    env = DelayedEffectEnv(horizon=args.horizon)
    policy = UniformPolicy()
    values = solve_exact_values(env, policy)
    rows: list[dict[str, float | int]] = []
    for num_rollouts in ROLLOUT_COUNTS:
        for seed in range(args.seed, args.seed + args.num_seeds):
            trajectories = sample_trajectories(env, policy, args.batch_size, seed)
            context = EstimatorContext(
                mdp=env, policy=policy, trajectories=trajectories
            )
            # Keep continuation simulation independent from the primary batch
            # stream while remaining exactly reproducible.
            mc_seed = MC_SEED_BASE + seed * MC_SEED_STRIDE
            estimator = MonteCarloAdvantage(
                num_rollouts=num_rollouts, seed=mc_seed
            )
            credits = estimator.estimate(context)
            squared = [
                (c - values.advantages[(s.timestep, s.state, s.action)]) ** 2
                for trajectory, row in zip(trajectories, credits, strict=True)
                for s, c in zip(trajectory.steps, row, strict=True)
            ]
            rows.append(
                {
                    "horizon": args.horizon,
                    "batch_size": args.batch_size,
                    "num_seeds": args.num_seeds,
                    "good_success_probability": env.good_success_probability,
                    "bad_success_probability": env.bad_success_probability,
                    "num_distractor_actions": env.num_distractor_actions,
                    "num_rollouts": num_rollouts,
                    "seed": seed,
                    "mc_seed": mc_seed,
                    "rmse": math.sqrt(sum(squared) / len(squared)),
                }
            )
        selected = [float(r["rmse"]) for r in rows if r["num_rollouts"] == num_rollouts]
        print(
            f"num_rollouts={num_rollouts}: "
            f"rmse={fmean(selected):.4f} +/- {pstdev(selected):.4f}"
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv_out}")

    errors = [
        fmean(float(r["rmse"]) for r in rows if r["num_rollouts"] == count)
        for count in ROLLOUT_COUNTS
    ]
    deviations = [
        pstdev(float(r["rmse"]) for r in rows if r["num_rollouts"] == count)
        for count in ROLLOUT_COUNTS
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ROLLOUT_COUNTS, errors, marker="o", label="mean RMSE")
    ax.fill_between(
        ROLLOUT_COUNTS,
        [m - d for m, d in zip(errors, deviations, strict=True)],
        [m + d for m, d in zip(errors, deviations, strict=True)],
        alpha=0.2,
        label="+/- 1 population std",
    )
    reference = [errors[0] / math.sqrt(k / ROLLOUT_COUNTS[0]) for k in ROLLOUT_COUNTS]
    ax.plot(
        ROLLOUT_COUNTS, reference, linestyle="--", label="1/sqrt(K) reference"
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("continuation samples per cached Q and V estimate")
    ax.set_ylabel("RMSE vs exact advantage")
    ax.set_title(
        f"MC convergence, DelayedEffectEnv(horizon={args.horizon})\n"
        f"batch={args.batch_size}, seeds={args.num_seeds}"
    )
    ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
