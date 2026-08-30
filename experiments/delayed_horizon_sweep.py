"""Delayed-effect horizon sweep — the first diagnostic figure (plan.md §11.1).

M2 exit condition: a reproducible plot of credit leaked onto zero-advantage
distractor actions, versus horizon.

Usage:
    pip install -e ".[experiments]"   # matplotlib, one-time
    python experiments/delayed_horizon_sweep.py --batch-size 1000 --seed 0
"""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, pstdev

from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    GiGPOStyle,
    GRPOStyleNormalized,
    OracleAdvantage,
    OutcomeBroadcast,
)
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Trajectory

HORIZONS = (2, 4, 8, 16, 32)
ESTIMATORS = (
    OracleAdvantage(),
    OutcomeBroadcast(),
    BatchCenteredBroadcast(),
    GRPOStyleNormalized(),
    GiGPOStyle(),
)


def _stable_value(value: object) -> object:
    """Keep committed numeric artifacts stable across runtimes/platforms."""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def mean_distractor_credit(
    trajectories: Sequence[Trajectory],
    credits: Sequence[Sequence[float]],
) -> float:
    """Mean |credit| over steps at t >= 1, where exact advantage is zero.

    The inline precursor of the M3 leakage metric (plan.md §10.4).
    """
    magnitudes = [
        abs(credit)
        for trajectory, row in zip(trajectories, credits, strict=True)
        for step, credit in zip(trajectory.steps, row, strict=True)
        if step.timestep >= 1
    ]
    return sum(magnitudes) / len(magnitudes)


def leakage_stats(
    trajectories: Sequence[Trajectory],
    credits: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Return per-step leakage, leaked credit per episode, and leakage ratio."""
    distractor = [
        abs(credit)
        for trajectory, row in zip(trajectories, credits, strict=True)
        for step, credit in zip(trajectory.steps, row, strict=True)
        if step.timestep >= 1
    ]
    total_abs = sum(abs(credit) for row in credits for credit in row)
    leaked = sum(distractor)
    return (
        leaked / len(distractor),
        leaked / len(trajectories),
        leaked / (total_abs + 1e-8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/delayed_leakage.png"))
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/delayed_horizon_sweep.csv"),
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

    rows: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        env = DelayedEffectEnv(horizon=horizon)
        policy = UniformPolicy()
        for seed in range(args.seed, args.seed + args.num_seeds):
            trajectories = sample_trajectories(env, policy, args.batch_size, seed)
            context = EstimatorContext(
                mdp=env, policy=policy, trajectories=trajectories
            )
            for estimator in ESTIMATORS:
                credits = estimator.estimate(context)
                per_step, per_episode, ratio = leakage_stats(trajectories, credits)
                rows.append(
                    {
                        "estimator": estimator.name,
                        "horizon": horizon,
                        "batch_size": args.batch_size,
                        "seed": seed,
                        "num_seeds": args.num_seeds,
                        "good_success_probability": env.good_success_probability,
                        "bad_success_probability": env.bad_success_probability,
                        "num_distractor_actions": env.num_distractor_actions,
                        "mean_abs_distractor_credit": per_step,
                        "total_abs_distractor_credit_per_episode": per_episode,
                        "leakage_ratio": ratio,
                    }
                )
        print(f"horizon={horizon}: done")

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fields = (
        ("mean_abs_distractor_credit", "mean |credit| per distractor"),
        (
            "total_abs_distractor_credit_per_episode",
            "total |credit| on distractors / episode",
        ),
        ("leakage_ratio", "fraction of |credit| on distractors"),
    )
    for estimator in ESTIMATORS:
        estimator_rows = [r for r in rows if r["estimator"] == estimator.name]
        for ax, (field, ylabel) in zip(axes, fields, strict=True):
            means = []
            deviations = []
            for horizon in HORIZONS:
                values = [
                    float(r[field])
                    for r in estimator_rows
                    if r["horizon"] == horizon
                ]
                means.append(fmean(values))
                deviations.append(pstdev(values))
            ax.plot(HORIZONS, means, marker="o", label=estimator.name)
            ax.fill_between(
                HORIZONS,
                [m - d for m, d in zip(means, deviations, strict=True)],
                [m + d for m, d in zip(means, deviations, strict=True)],
                alpha=0.12,
            )
            ax.set_xlabel("horizon")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Credit leakage vs horizon (batch={args.batch_size}, "
        f"seeds={args.num_seeds})"
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
