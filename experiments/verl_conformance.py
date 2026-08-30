"""Recovery diagnostic on verl's actual advantage implementations.

Same protocol as recovery_diagnostic.py, but the estimators under test are
verl's own functions (via agent_credit_bench.integrations.verl), not the
suite's "-style" reimplementations. Requires the [verl] extra; no matplotlib
needed — prints the table and writes it as CSV.

Usage:
    python experiments/verl_conformance.py --batch-size 2000 --seed 0
"""

import argparse
import csv
from importlib.metadata import version
from pathlib import Path

from agent_credit_bench.envs.recovery import RecoveryEnv
from agent_credit_bench.estimators import EstimatorContext, OracleAdvantage
from agent_credit_bench.integrations.verl import (
    VerlGAE,
    VerlGRPO,
    VerlReinforcePlusPlus,
    VerlRLOO,
)
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

ESTIMATORS = (
    OracleAdvantage(),
    VerlGRPO(),
    VerlGRPO(norm_adv_by_std=False),
    VerlRLOO(),
    VerlReinforcePlusPlus(),
    VerlGAE(critic="exact", lam=1.0),
    VerlGAE(critic="exact", lam=0.0),
    VerlGAE(critic="zero", lam=1.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recover-success", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("results/verl_recovery.csv"))
    args = parser.parse_args()
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2")

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
    if not recovered:
        raise RuntimeError("sample contained no successful BAD -> RECOVER path")
    print(f"{len(recovered)} successful BAD -> RECOVER trajectories")

    rows = []
    for estimator in ESTIMATORS:
        credits = estimator.estimate(context)
        bad = [credits[i][0] for i in recovered]
        recover = [credits[i][1] for i in recovered]
        rows.append(
            {
                "environment": "recovery",
                "recover_success_probability": args.recover_success,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "num_recovered": len(recovered),
                "framework": (
                    "agent-credit-bench"
                    if estimator.name == "oracle_advantage"
                    else "verl"
                ),
                "framework_version": (
                    "source-tree"
                    if estimator.name == "oracle_advantage"
                    else version("verl")
                ),
                "estimator": estimator.name,
                "mean_credit_bad": sum(bad) / len(bad),
                "mean_credit_recover": sum(recover) / len(recover),
                "both_positive_fraction_on_recovered": sum(
                    b > 0 and r > 0 for b, r in zip(bad, recover, strict=True)
                )
                / len(bad),
            }
        )

    print(f"{'estimator':<28}{'credit(BAD)':>14}{'credit(RECOVER)':>17}{'both>0':>9}")
    for row in rows:
        print(
            f"{row['estimator']:<28}{row['mean_credit_bad']:>14.4f}"
            f"{row['mean_credit_recover']:>17.4f}"
            f"{row['both_positive_fraction_on_recovered']:>9.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
