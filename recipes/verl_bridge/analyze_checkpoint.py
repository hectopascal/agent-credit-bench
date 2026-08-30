"""Score credit estimators on real LLM rollouts from bridge episode logs.

Reads episodes.jsonl written during training (or offline play), measures the
finite-sample Markov projection ``pi_hat(action | timestep, state)``, solves
exact values for that fitted projection, and reports every estimator's credit
quality on the logged trajectories. This is not an exact reconstruction of a
history-conditioned LLM policy. Only needs the core suite; verl estimator rows
appear automatically when verl is installed.

Usage:
    python analyze_checkpoint.py episodes.jsonl [more.jsonl ...] \
        [--last 2000] [--out metrics.csv]

--last N analyzes only the newest N episodes — the sliding window that
approximates "the current checkpoint's policy" when one file spans a run.
N must be a positive integer.
"""

import argparse
import csv
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    GiGPOStyle,
    GRPOStyleNormalized,
    OracleAdvantage,
    OutcomeBroadcast,
    TurnLOO,
)
from agent_credit_bench.integrations.bridge import (
    EmpiricalTabularPolicy,
    load_episodes,
    make_env,
    trajectory_from_record,
)
from agent_credit_bench.metrics import (
    centered_rmse,
    leakage,
    per_turn_stats,
    rmse,
    sign_accuracy,
    spearman,
)
from agent_credit_bench.oracle import solve_exact_values


def positive_integer_argument(value: str) -> int:
    """argparse type for strictly positive window sizes."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_estimators() -> list:
    estimators = [
        OracleAdvantage(),
        OutcomeBroadcast(),
        BatchCenteredBroadcast(),
        TurnLOO(),
        GRPOStyleNormalized(),
        GiGPOStyle(),
    ]
    # the integrations import lazily, so probe for the frameworks themselves.
    if importlib.util.find_spec("verl") is None:
        print("(verl not installed — skipping verl estimator rows)")
    else:
        from agent_credit_bench.integrations.verl import (
            VerlGAE,
            VerlGRPO,
            VerlRLOO,
        )

        estimators += [VerlGRPO(), VerlRLOO(), VerlGAE(critic="exact", lam=0.0)]
    if importlib.util.find_spec("trl") is None:
        print("(trl not installed — skipping trl estimator rows)")
    else:
        from agent_credit_bench.integrations.trl import TrlGRPO, TrlRLOO

        estimators += [TrlGRPO(), TrlRLOO()]
    if importlib.util.find_spec("openrlhf") is None:
        print("(openrlhf not installed — skipping openrlhf estimator rows)")
    else:
        from agent_credit_bench.integrations.openrlhf import (
            OpenRLHFGAE,
            OpenRLHFOutcome,
        )

        estimators += [
            OpenRLHFOutcome("group_norm"),
            OpenRLHFOutcome("rloo"),
            OpenRLHFGAE(critic="exact", lam=0.0),
        ]
    return estimators


def environment_spec(records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Return the one environment specification shared by all records."""
    if not records:
        raise SystemExit("no episodes found")
    env_names = {record["env"] for record in records}
    if len(env_names) > 1:
        raise SystemExit(f"episodes mix environments {sorted(env_names)}")
    env_name = env_names.pop()

    params_by_record: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for index, record in enumerate(records):
        raw_params = record.get("env_params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, Mapping):
            raise SystemExit(
                f"episode {index} has non-object env_params {raw_params!r}"
            )
        params = dict(raw_params)
        try:
            fingerprint = json.dumps(
                params,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise SystemExit(
                f"episode {index} has invalid JSON env_params: {error}"
            ) from error
        params_by_record.append(params)
        fingerprints.append(fingerprint)

    env_params = params_by_record[0]
    for index, fingerprint in enumerate(fingerprints[1:], start=1):
        if fingerprint != fingerprints[0]:
            raise SystemExit(
                f"episodes mix env_params for {env_name!r}: "
                f"record 0 has {env_params!r}, record {index} has "
                f"{params_by_record[index]!r}"
            )
    return env_name, env_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--last", type=positive_integer_argument, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load_episodes(args.paths)
    if args.last is not None:
        records = records[-args.last :]
    if not records:
        raise SystemExit("no episodes found")
    env_name, env_params = environment_spec(records)
    mdp = make_env(env_name, **env_params)

    turns = [turn for record in records for turn in record["turns"]]
    parsed_rate = sum(turn["parsed"] for turn in turns) / len(turns)
    trajectories = tuple(trajectory_from_record(record) for record in records)
    mean_return = sum(t.total_return for t in trajectories) / len(trajectories)
    print(
        f"{len(trajectories)} episodes, {len(turns)} turns, "
        f"parsed rate {parsed_rate:.3f}, mean return {mean_return:.3f}"
    )
    if parsed_rate < 0.9:
        print("WARNING: low parsed rate — projection includes fallback actions")

    policy = EmpiricalTabularPolicy.from_trajectories(trajectories)
    exact = solve_exact_values(mdp, policy)
    print(
        "oracle target: exact DP for the finite-sample Markov projection "
        "pi_hat(a | timestep, state), not the history-conditioned LLM policy"
    )
    print("\nfinite-sample Markov projection (visit counts):")
    for t in range(mdp.horizon):
        for state in mdp.states_at(t):
            count = policy.visit_count(t, state)
            if count == 0:
                continue
            probs = policy.action_probabilities(t, state, mdp.actions(t, state))
            rendered = ", ".join(f"{a}: {p:.3f}" for a, p in probs.items())
            print(f"  t={t} state={state!r} n={count}: {rendered}")

    exact_flat = [
        exact.advantages[(s.timestep, s.state, s.action)]
        for trajectory in trajectories
        for s in trajectory.steps
    ]
    keys = [
        (s.timestep, s.state)
        for trajectory in trajectories
        for s in trajectory.steps
    ]
    timesteps = [
        s.timestep for trajectory in trajectories for s in trajectory.steps
    ]

    context = EstimatorContext(mdp=mdp, policy=policy, trajectories=trajectories)
    estimators = build_estimators()
    header = (
        f"\n{'estimator':<28}{'rmse':>8}{'c-rmse':>8}{'sign':>7}"
        f"{'leak':>7}{'spear':>7}"
    )
    print(header)
    rows = []
    for estimator in estimators:
        credits = estimator.estimate(context)
        estimated = [credit for row in credits for credit in row]
        signs = sign_accuracy(estimated, exact_flat)
        row = {
            "estimator": estimator.name,
            "rmse": rmse(estimated, exact_flat),
            "centered_rmse": centered_rmse(estimated, exact_flat, keys).value,
            "sign_accuracy": signs.accuracy,
            "leakage_ratio": leakage(estimated, exact_flat).ratio,
            "spearman": spearman(estimated, exact_flat),
        }
        rows.append(row)
        signs_known = row["sign_accuracy"] is not None
        sign_text = f"{row['sign_accuracy']:5.2f}" if signs_known else "  n/a"
        print(
            f"{row['estimator']:<28}{row['rmse']:>8.3f}"
            f"{row['centered_rmse']:>8.3f}{sign_text:>7}"
            f"{row['leakage_ratio']:>7.2f}{row['spearman']:>7.2f}"
        )
        if estimator.name == "oracle_advantage":
            bias = per_turn_stats(timesteps, estimated, exact_flat)
            worst = max(abs(stat.bias) for stat in bias.values())
            assert worst < 1e-9, "oracle disagrees with itself — bug"

    if args.out:
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
