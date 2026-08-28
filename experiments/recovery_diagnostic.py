"""Recovery credit diagnostic (plan.md §11.2, M4 exit condition).

Conditions on successful BAD -> RECOVER trajectories and reports the average
credit each estimator assigns to the mistake and to the repair. A useful
estimator separates the two signs; a broadcast estimator praises both.

Usage:
    python experiments/recovery_diagnostic.py --batch-size 2000 --seed 0
"""

import argparse
from pathlib import Path

from agent_credit_bench.envs.recovery import RecoveryEnv
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

ESTIMATORS = (
    OracleAdvantage(),
    OutcomeBroadcast(),
    BatchCenteredBroadcast(),
    GRPOStyleNormalized(),
    GiGPOStyle(),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recover-success", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("results/recovery_credit.png"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'matplotlib missing — run: pip install -e ".[experiments]"'
        ) from None

    env = RecoveryEnv(recover_success_probability=args.recover_success)
    policy = UniformPolicy()
    trajectories = sample_trajectories(env, policy, args.batch_size, args.seed)
    context = EstimatorContext(mdp=env, policy=policy, trajectories=trajectories)

    recovered = [
        i
        for i, trajectory in enumerate(trajectories)
        if len(trajectory.steps) == 2
        and trajectory.steps[0].action == "BAD"
        and trajectory.steps[1].action == "RECOVER"
        and trajectory.total_return == 1.0
    ]
    print(f"{len(recovered)} successful BAD -> RECOVER trajectories")

    mean_bad: dict[str, float] = {}
    mean_recover: dict[str, float] = {}
    both_praised: dict[str, float] = {}
    for estimator in ESTIMATORS:
        credits = estimator.estimate(context)
        bad = [credits[i][0] for i in recovered]
        recover = [credits[i][1] for i in recovered]
        mean_bad[estimator.name] = sum(bad) / len(bad)
        mean_recover[estimator.name] = sum(recover) / len(recover)
        both_praised[estimator.name] = sum(
            b > 0 and r > 0 for b, r in zip(bad, recover, strict=True)
        ) / len(bad)

    print(f"{'estimator':<28}{'credit(BAD)':>14}{'credit(RECOVER)':>17}{'both>0':>9}")
    for name in mean_bad:
        print(
            f"{name:<28}{mean_bad[name]:>14.4f}"
            f"{mean_recover[name]:>17.4f}{both_praised[name]:>9.2f}"
        )

    names = list(mean_bad)
    positions = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [p - width / 2 for p in positions],
        [mean_bad[n] for n in names],
        width,
        label="BAD (exact advantage < 0)",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [mean_recover[n] for n in names],
        width,
        label="RECOVER (exact advantage > 0)",
    )
    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("mean credit on successful BAD -> RECOVER trajectories")
    ax.set_title(
        f"Recovery diagnostic (batch={args.batch_size}, seed={args.seed})"
    )
    ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
