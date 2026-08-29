"""Recovery diagnostic across every installed framework integration.

Same protocol as recovery_diagnostic.py / verl_conformance.py, but the table
mixes rows from whichever of verl, TRL, and OpenRLHF are importable — skipped
frameworks are reported, not errors, so the script runs usefully in any
environment (no single venv holds all three: OpenRLHF is Linux-only).

Usage:
    python experiments/cross_framework_conformance.py --batch-size 2000
"""

import argparse
import csv
import importlib.util
from pathlib import Path

from agent_credit_bench.envs.recovery import RecoveryEnv
from agent_credit_bench.estimators import EstimatorContext, OracleAdvantage
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories


def build_estimators() -> tuple[list, list[str]]:
    estimators: list = [OracleAdvantage()]
    skipped: list[str] = []
    if importlib.util.find_spec("verl") is not None:
        from agent_credit_bench.integrations.verl import (
            VerlGAE,
            VerlGRPO,
            VerlRLOO,
        )

        estimators += [
            VerlGRPO(),
            VerlRLOO(),
            VerlGAE(critic="exact", lam=1.0),
            VerlGAE(critic="exact", lam=0.0),
        ]
    else:
        skipped.append("verl")
    if importlib.util.find_spec("trl") is not None:
        from agent_credit_bench.integrations.trl import TrlGRPO, TrlRLOO

        estimators += [TrlGRPO(), TrlGRPO(scale_rewards="none"), TrlRLOO()]
    else:
        skipped.append("trl")
    if importlib.util.find_spec("openrlhf") is not None:
        from agent_credit_bench.integrations.openrlhf import (
            OpenRLHFGAE,
            OpenRLHFOutcome,
        )

        estimators += [
            OpenRLHFOutcome("group_norm"),
            OpenRLHFOutcome("rloo"),
            OpenRLHFOutcome("reinforce_baseline"),
            OpenRLHFGAE(critic="exact", lam=1.0),
            OpenRLHFGAE(critic="exact", lam=0.0),
        ]
    else:
        skipped.append("openrlhf")
    return estimators, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recover-success", type=float, default=1.0)
    parser.add_argument(
        "--out", type=Path, default=Path("results/cross_framework_recovery.csv")
    )
    args = parser.parse_args()

    estimators, skipped = build_estimators()
    if skipped:
        print(f"not installed (rows omitted): {', '.join(skipped)}")

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

    rows = []
    for estimator in estimators:
        credits = estimator.estimate(context)
        bad = [credits[i][0] for i in recovered]
        recover = [credits[i][1] for i in recovered]
        rows.append(
            {
                "estimator": estimator.name,
                "mean_credit_bad": sum(bad) / len(bad),
                "mean_credit_recover": sum(recover) / len(recover),
                "both_praised_fraction": sum(
                    b > 0 and r > 0 for b, r in zip(bad, recover, strict=True)
                )
                / len(bad),
            }
        )

    print(f"{'estimator':<30}{'credit(BAD)':>14}{'credit(RECOVER)':>17}{'both>0':>9}")
    for row in rows:
        print(
            f"{row['estimator']:<30}{row['mean_credit_bad']:>14.4f}"
            f"{row['mean_credit_recover']:>17.4f}{row['both_praised_fraction']:>9.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
