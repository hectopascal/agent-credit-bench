"""Recovery diagnostic across every installed framework integration.

Same protocol as recovery_diagnostic.py / verl_conformance.py, but the table
mixes rows from whichever of verl, TRL, and OpenRLHF are importable. Missing
frameworks are reported and omitted; OpenRLHF rows require Linux x86_64.

Usage:
    python experiments/cross_framework_conformance.py --batch-size 2000

When exact framework pins conflict, run once per environment to separate CSVs,
then merge them without recomputing:

    python experiments/cross_framework_conformance.py \
        --merge-input /tmp/verl_trl.csv /tmp/openrlhf.csv
"""

import argparse
import csv
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import agent_credit_bench
from agent_credit_bench.envs.recovery import RecoveryEnv
from agent_credit_bench.estimators import EstimatorContext, OracleAdvantage
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

ESTIMATOR_ORDER = (
    "oracle_advantage",
    "verl_grpo",
    "verl_rloo",
    "verl_gae_exact_lam1",
    "verl_gae_exact_lam0",
    "trl_grpo",
    "trl_dr_grpo",
    "trl_rloo",
    "openrlhf_group_norm",
    "openrlhf_rloo",
    "openrlhf_reinforce_baseline",
    "openrlhf_gae_exact_lam1",
    "openrlhf_gae_exact_lam0",
)


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "source-tree"


def suite_version() -> str:
    """Do not label a PYTHONPATH checkout with an unrelated installed version."""
    try:
        installed = distribution("agent-credit-bench")
    except PackageNotFoundError:
        return "source-tree"
    installed_init = installed.locate_file("agent_credit_bench/__init__.py")
    if Path(agent_credit_bench.__file__).resolve() != Path(installed_init).resolve():
        return "source-tree"
    return installed.version


def estimator_provenance(name: str) -> tuple[str, str]:
    if name.startswith("verl_"):
        return "verl", package_version("verl")
    if name.startswith("trl_"):
        return "trl", package_version("trl")
    if name.startswith("openrlhf_"):
        return "openrlhf", package_version("openrlhf")
    return "agent-credit-bench", suite_version()


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


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty conformance table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def merge_rows(paths: list[Path]) -> list[dict[str, str]]:
    """Merge separately validated framework rows with one shared protocol."""
    combined: dict[str, dict[str, str]] = {}
    fields: list[str] | None = None
    protocol: tuple[str, ...] | None = None
    protocol_fields = (
        "environment",
        "recover_success_probability",
        "batch_size",
        "seed",
        "num_recovered",
    )
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no header")
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise ValueError(f"{path} has a different CSV schema")
            for row in reader:
                row_protocol = tuple(row[field] for field in protocol_fields)
                if protocol is None:
                    protocol = row_protocol
                elif row_protocol != protocol:
                    raise ValueError(f"{path} uses a different experiment protocol")
                previous = combined.get(row["estimator"])
                if previous is not None and previous != row:
                    raise ValueError(
                        f"conflicting rows for estimator {row['estimator']!r}"
                    )
                combined[row["estimator"]] = row

    order = {name: index for index, name in enumerate(ESTIMATOR_ORDER)}
    return sorted(
        combined.values(),
        key=lambda row: (order.get(row["estimator"], len(order)), row["estimator"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recover-success", type=float, default=1.0)
    parser.add_argument(
        "--out", type=Path, default=Path("results/cross_framework_recovery.csv")
    )
    parser.add_argument(
        "--merge-input",
        nargs="+",
        type=Path,
        help="merge partial CSVs produced in separately pinned environments",
    )
    args = parser.parse_args()
    if args.merge_input:
        write_rows(args.out, merge_rows(args.merge_input))
        return
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2")

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
    if not recovered:
        raise RuntimeError("sample contained no successful BAD -> RECOVER path")
    print(f"{len(recovered)} successful BAD -> RECOVER trajectories")

    rows = []
    for estimator in estimators:
        credits = estimator.estimate(context)
        bad = [credits[i][0] for i in recovered]
        recover = [credits[i][1] for i in recovered]
        framework, framework_version = estimator_provenance(estimator.name)
        rows.append(
            {
                "environment": "recovery",
                "recover_success_probability": args.recover_success,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "num_recovered": len(recovered),
                "framework": framework,
                "framework_version": framework_version,
                "estimator": estimator.name,
                "mean_credit_bad": sum(bad) / len(bad),
                "mean_credit_recover": sum(recover) / len(recover),
                "both_positive_fraction_on_recovered": sum(
                    b > 0 and r > 0 for b, r in zip(bad, recover, strict=True)
                )
                / len(bad),
            }
        )

    print(f"{'estimator':<30}{'credit(BAD)':>14}{'credit(RECOVER)':>17}{'both>0':>9}")
    for row in rows:
        print(
            f"{row['estimator']:<30}{row['mean_credit_bad']:>14.4f}"
            f"{row['mean_credit_recover']:>17.4f}"
            f"{row['both_positive_fraction_on_recovered']:>9.2f}"
        )

    write_rows(args.out, rows)


if __name__ == "__main__":
    main()
